"""Texture maps & atlas: a generalised Flora.

Extracts any channel of any material (Diffuse, Opacity, Roughness, Metallic, Normal,
Emission, AO) by baking it with Cycles, so procedural and mixed node trees work too,
not only "one image in Base Color". Keeps Flora's foliage helpers (edge padding of
colour into transparent pixels, hard / inverted opacity) and can pack channels
(Diffuse+Opacity → RGBA, AO/Roughness/Metallic → ORM).

Two kinds of output:
  * per-material maps — each material is baked on a temporary 0–1 plane, which
    reproduces its own texture space;
  * an atlas for a mesh — either Flora's grid (each material gets a cell, the UVs are
    moved into it) or "repack" (UV islands are packed into a new UV map and baked from
    the mesh, which also handles tiling UVs outside 0–1).
"""

import math
import os

import bpy
import numpy as np
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from . import common as C

# id: (label, default suffix, principled input names, sRGB?, bake kind)
CHANNELS = {
    "DIFFUSE": ("Diffuse", "_Diffuse", ("Base Color",), True, "EMIT"),
    "OPACITY": ("Opacity", "_Opacity", ("Alpha",), False, "EMIT"),
    "ROUGHNESS": ("Roughness", "_Roughness", ("Roughness",), False, "EMIT"),
    "METALLIC": ("Metallic", "_Metallic", ("Metallic",), False, "EMIT"),
    "NORMAL": ("Normal", "_Normal", (), False, "NORMAL"),
    "EMISSION": ("Emission", "_Emission", ("Emission Color", "Emission"), True, "EMIT"),
    "AO": ("AO", "_AO", (), False, "AO"),
}
ORDER = ("DIFFUSE", "OPACITY", "ROUGHNESS", "METALLIC", "NORMAL", "EMISSION", "AO")
DEFAULT_FILL = {
    "DIFFUSE": (0.5, 0.5, 0.5), "OPACITY": (1, 1, 1), "ROUGHNESS": (0.5, 0.5, 0.5),
    "METALLIC": (0, 0, 0), "NORMAL": (0.5, 0.5, 1.0), "EMISSION": (0, 0, 0), "AO": (1, 1, 1),
}
EXT = {"PNG": ".png", "TARGA": ".tga", "JPEG": ".jpg", "TIFF": ".tif"}
TEMP = "tk_bake_tmp"


# ---------------------------------------------------------------------------
# Finding where a channel comes from in a node tree
# ---------------------------------------------------------------------------

def ensure_nodes(mat):
    """Make sure *mat* has an active node tree. Returns a callable that undoes it."""
    if mat is None:
        return lambda: None
    used = getattr(mat, "use_nodes", True)
    if mat.node_tree is not None and used:
        return lambda: None
    try:
        mat.use_nodes = True
    except Exception:
        pass

    def undo():
        try:
            mat.use_nodes = used
        except Exception:
            pass
    return undo


def output_node(nt):
    outs = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    for n in outs:
        if n.is_active_output:
            return n
    return outs[0] if outs else None


def _linked_from(sock):
    if sock is None or not sock.is_linked:
        return None
    link = sock.links[0]
    node = link.from_node
    while node.bl_idname == "NodeReroute" and node.inputs[0].is_linked:
        link = node.inputs[0].links[0]
        node = link.from_node
    return link


def analyse(mat):
    """Return dict: shader node, opacity info. Handles Principled, other BSDFs and the
    foliage setup Mix Shader(Transparent, BSDF) with the mask in Fac."""
    info = {"shader": None, "opacity": None}
    if mat is None or mat.node_tree is None:
        return info
    nt = mat.node_tree
    out = output_node(nt)
    link = _linked_from(out.inputs["Surface"]) if out else None
    node = link.from_node if link else None
    if node is not None and node.bl_idname == "ShaderNodeMixShader":
        l1 = _linked_from(node.inputs[1])
        l2 = _linked_from(node.inputs[2])
        n1 = l1.from_node if l1 else None
        n2 = l2.from_node if l2 else None
        if n1 is not None and n1.bl_idname == "ShaderNodeBsdfTransparent":
            info["opacity"] = ("FAC", node.inputs[0], False)   # opacity = Fac
            node = n2
        elif n2 is not None and n2.bl_idname == "ShaderNodeBsdfTransparent":
            info["opacity"] = ("FAC", node.inputs[0], True)    # opacity = 1 - Fac
            node = n1
    if node is None or node.bl_idname == "ShaderNodeMixShader":
        node = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), node)
    info["shader"] = node
    return info


def _shader_input(shader, names):
    if shader is None:
        return None
    for nm in names:
        s = shader.inputs.get(nm)
        if s is not None:
            return s
    return None


def channel_source(mat, ch, info=None):
    """('SOCKET', output_socket, invert) or ('VALUE', (r,g,b)) for an EMIT channel."""
    info = info or analyse(mat)
    shader = info["shader"]
    if ch == "OPACITY" and info["opacity"] is not None:
        _k, fac, inv = info["opacity"]
        if fac.is_linked:
            return ("SOCKET", _linked_from(fac).from_socket, inv)
        v = float(fac.default_value)
        v = 1.0 - v if inv else v
        return ("VALUE", (v, v, v))
    names = CHANNELS[ch][2]
    if ch == "DIFFUSE":
        names = names + ("Color",)
    sock = _shader_input(shader, names)
    if sock is not None:
        link = _linked_from(sock)
        if link is not None:
            return ("SOCKET", link.from_socket, False)
        dv = sock.default_value
        try:
            rgb = (dv[0], dv[1], dv[2])
        except TypeError:
            rgb = (dv, dv, dv)
        if ch == "EMISSION":
            st = shader.inputs.get("Emission Strength")
            k = st.default_value if st is not None and not st.is_linked else 1.0
            rgb = tuple(c * k for c in rgb)
        return ("VALUE", rgb)
    if ch == "DIFFUSE" and mat is not None:
        c = mat.diffuse_color
        return ("VALUE", (c[0], c[1], c[2]))
    if ch == "OPACITY":
        return ("VALUE", (1.0, 1.0, 1.0))
    return ("VALUE", DEFAULT_FILL[ch])


def material_resolution(mat, fallback):
    w = h = 0
    if mat is not None and mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeTexImage" and n.image and n.image.size[0] > 0:
                w = max(w, n.image.size[0])
                h = max(h, n.image.size[1])
    if w == 0:
        return fallback, fallback
    return w, h


def uv_names_used(mat):
    names = []
    if mat is not None and mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            nm = getattr(n, "uv_map", "")
            if nm and nm not in names:
                names.append(nm)
    return names


def missing_images(mat):
    bad = []
    if mat is not None and mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeTexImage" and n.image is not None and n.image.size[0] == 0:
                bad.append(n.image.name)
    return bad


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------

def new_image(name, w, h, ch, alpha=False):
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, w, h, alpha=alpha)
    fill = DEFAULT_FILL[ch]
    img.generated_color = (fill[0], fill[1], fill[2], 1.0)
    if not CHANNELS[ch][3]:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    return img


class _MatPatch:
    """Temporarily rewires one material for baking one channel, then restores it."""

    def __init__(self, mat, ch, image):
        self.mat = mat
        self.nt = mat.node_tree
        self.nodes = []
        self.prev_active_out = None
        self.prev_active = self.nt.nodes.active
        kind = CHANNELS[ch][4]
        img_node = None
        if image is not None:  # None: only rewire (source of a selected-to-active bake)
            img_node = self._add("ShaderNodeTexImage")
            img_node.image = image
            img_node.location = (-2000, 2000)
        if kind == "EMIT":
            src = channel_source(mat, ch)
            em = self._add("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            if src[0] == "SOCKET":
                sock = src[1]
                if src[2]:
                    inv = self._add("ShaderNodeMath")
                    inv.operation = "SUBTRACT"
                    inv.inputs[0].default_value = 1.0
                    self.nt.links.new(sock, inv.inputs[1])
                    sock = inv.outputs[0]
                self.nt.links.new(sock, em.inputs["Color"])
            else:
                v = src[1]
                em.inputs["Color"].default_value = (v[0], v[1], v[2], 1.0)
            out = self._add("ShaderNodeOutputMaterial")
            out.target = "ALL"
            self.prev_active_out = output_node(self.nt)
            self.nt.links.new(em.outputs[0], out.inputs["Surface"])
            out.is_active_output = True
        if img_node is not None:
            for n in self.nt.nodes:
                n.select = False
            img_node.select = True
            self.nt.nodes.active = img_node

    def _add(self, idname):
        n = self.nt.nodes.new(idname)
        n[TEMP] = 1
        self.nodes.append(n)
        return n

    def restore(self):
        for n in self.nodes:
            try:
                self.nt.nodes.remove(n)
            except Exception:
                pass
        if self.prev_active_out is not None:
            try:
                self.prev_active_out.is_active_output = True
            except Exception:
                pass
        try:
            self.nt.nodes.active = self.prev_active
        except Exception:
            pass


class _SceneForBake:
    """Cycles + selection set up for baking one object; restored on exit."""

    def __init__(self, context, obj, samples=1, extra=()):
        self.ctx = context
        self.obj = obj
        self.samples = samples
        self.extra = list(extra)

    def __enter__(self):
        sc = self.ctx.scene
        vl = self.ctx.view_layer
        self.engine = sc.render.engine
        self.cyc_samples = sc.cycles.samples
        self.sel = C.selected_objects(vl)
        self.active = vl.objects.active
        self.hidden = self.obj.hide_get()
        self.hide_render = self.obj.hide_render
        sc.render.engine = "CYCLES"
        sc.cycles.samples = self.samples
        C.deselect_all(vl)
        self.obj.hide_set(False)
        self.obj.hide_render = False
        self.extra_hidden = [(o, o.hide_get(), o.hide_render) for o in self.extra]
        for o in self.extra:
            o.hide_set(False)
            o.hide_render = False
            o.select_set(True)
        self.obj.select_set(True)
        vl.objects.active = self.obj
        return self

    def __exit__(self, *a):
        sc = self.ctx.scene
        vl = self.ctx.view_layer
        sc.render.engine = self.engine
        sc.cycles.samples = self.cyc_samples
        try:
            self.obj.select_set(False)
            self.obj.hide_set(self.hidden)
            self.obj.hide_render = self.hide_render
        except ReferenceError:
            pass
        for o, h, hr in self.extra_hidden:
            try:
                o.select_set(False)
                o.hide_set(h)
                o.hide_render = hr
            except ReferenceError:
                pass
        for o in self.sel:
            try:
                o.select_set(True)
            except ReferenceError:
                pass
        try:
            vl.objects.active = self.active
        except ReferenceError:
            pass


def bake_channel(context, obj, ch, image, margin=8, uv_layer="", ao_samples=32):
    """Bake one channel of every material of *obj* into *image*."""
    mats = []
    for slot in obj.material_slots:
        if slot.material is not None and slot.material not in mats:
            mats.append(slot.material)
    undos = [ensure_nodes(m) for m in mats]
    patches = []
    kind = CHANNELS[ch][4]
    try:
        for m in mats:
            patches.append(_MatPatch(m, ch, image))
        samples = ao_samples if kind == "AO" else 1
        with _SceneForBake(context, obj, samples):
            kwargs = dict(type=kind, margin=margin, use_clear=True, target="IMAGE_TEXTURES",
                          use_selected_to_active=False, use_cage=False, uv_layer=uv_layer)
            if kind == "NORMAL":
                kwargs.update(normal_space="TANGENT")
            try:
                bpy.ops.object.bake(margin_type="EXTEND", **kwargs)
            except TypeError:
                bpy.ops.object.bake(**kwargs)
    finally:
        for p in patches:
            p.restore()
        for u in undos:
            u()


# ---------------------------------------------------------------------------
# Pixel helpers
# ---------------------------------------------------------------------------

def image_array(img):
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(h, w, 4)


def set_image_array(img, arr):
    img.pixels.foreach_set(np.clip(arr, 0.0, 1.0).astype(np.float32).ravel())
    img.update()


def bleed_colors(rgb, mask, iterations):
    """Flora's edge padding: grow colour of opaque pixels into transparent ones."""
    rgb = rgb.copy()
    valid = mask.copy()
    if not valid.any():
        return rgb
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for _ in range(iterations):
        if valid.all():
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros(valid.shape, np.float32)
        vf = valid.astype(np.float32)
        for dy, dx in shifts:
            acc += np.roll(rgb * vf[..., None], (dy, dx), axis=(0, 1))
            cnt += np.roll(vf, (dy, dx), axis=(0, 1))
        grow = (~valid) & (cnt > 0)
        if not grow.any():
            break
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        valid |= grow
    if not valid.all():
        rgb[~valid] = rgb[mask].mean(axis=0)
    return rgb


def postprocess(arrays, p):
    """Flora options on baked arrays (dict ch -> HxWx4). Returns list of warnings."""
    warns = []
    op = arrays.get("OPACITY")
    if op is not None:
        a = op[..., 0]
        if a.min() >= 0.999:
            warns.append("нет прозрачности")
        else:
            if p.bleed and "DIFFUSE" in arrays:
                d = arrays["DIFFUSE"]
                d[..., :3] = bleed_colors(d[..., :3], a >= p.threshold, p.bleed_px)
            if p.binary_mask:
                a = (a >= p.threshold).astype(np.float32)
            if p.invert:
                a = 1.0 - a
            op[..., 0] = op[..., 1] = op[..., 2] = a
    return warns


def save_array(name, arr, path, fmt, srgb, alpha=False):
    h, w = arr.shape[:2]
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, w, h, alpha=alpha)
    if not srgb:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    rgba = arr.copy()
    if not alpha:
        rgba[..., 3] = 1.0
    set_image_array(img, rgba)
    if path:
        img.filepath_raw = path
        img.file_format = fmt
        img.save()
        img.filepath = path
        img.source = "FILE"
        img.reload()
        if not srgb:
            try:
                img.colorspace_settings.name = "Non-Color"
            except TypeError:
                pass
    else:
        try:
            img.pack()
        except RuntimeError:
            pass
    return img


def packed_outputs(arrays, p):
    """Return list of (suffix, RGBA array, srgb, has_alpha)."""
    out = []
    if p.pack_rgba and "DIFFUSE" in arrays:
        d = arrays["DIFFUSE"].copy()
        op = arrays.get("OPACITY")
        d[..., 3] = op[..., 0] if op is not None else 1.0
        out.append(("_DiffuseOpacity", d, True, True))
    if p.pack_orm and any(c in arrays for c in ("AO", "ROUGHNESS", "METALLIC")):
        shape = next(arrays[c] for c in ("AO", "ROUGHNESS", "METALLIC", "DIFFUSE") if c in arrays).shape
        orm = np.ones(shape, np.float32)
        for i, c in enumerate(("AO", "ROUGHNESS", "METALLIC")):
            if c in arrays:
                orm[..., i] = arrays[c][..., 0]
            elif c == "METALLIC":
                orm[..., i] = 0.0
            elif c == "ROUGHNESS":
                orm[..., i] = 0.5
        out.append(("_ORM", orm, False, False))
    return out


# ---------------------------------------------------------------------------
# Per-material extraction on a temporary plane
# ---------------------------------------------------------------------------

def make_plane(context, mat):
    me = bpy.data.meshes.new("TK_bake_plane")
    me.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
    names = ["UVMap"] + [n for n in uv_names_used(mat) if n != "UVMap"]
    uvco = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for nm in names[:8]:
        uv = me.uv_layers.new(name=nm)
        for i, d in enumerate(uv.data):
            d.uv = uvco[i]
    me.materials.append(mat)
    ob = bpy.data.objects.new("TK_bake_plane", me)
    context.scene.collection.objects.link(ob)
    return ob


def remove_object(ob):
    me = ob.data
    bpy.data.objects.remove(ob)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def bake_material_arrays(context, mat, channels, w, h):
    """Bake *channels* of *mat* at w×h. Returns dict ch -> array (H, W, 4)."""
    ob = make_plane(context, mat)
    arrays = {}
    try:
        for ch in channels:
            img = new_image(f"TK_tmp_{ch}", w, h, ch)
            try:
                bake_channel(context, ob, ch, img, margin=2)
                arrays[ch] = image_array(img).copy()
            finally:
                bpy.data.images.remove(img)
    finally:
        remove_object(ob)
    return arrays


# ---------------------------------------------------------------------------
# Material wiring
# ---------------------------------------------------------------------------

def set_transparency_mode(mat):
    if hasattr(mat, "surface_render_method"):
        try:
            mat.surface_render_method = "DITHERED"
        except Exception:
            pass
    for attr in ("blend_method", "shadow_method"):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, "CLIP")
            except Exception:
                pass


def wire_images(mat, images, uv_name=None, x=-450, y=0):
    """Connect channel images (dict ch -> Image) to the material's Principled BSDF."""
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return False
    uvn = None
    if uv_name:
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = uv_name
        uvn.location = (bsdf.location.x + x - 300, bsdf.location.y)
    row = 0
    for ch in ORDER:
        img = images.get(ch)
        if img is None:
            continue
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.label = tex.name = "TK_" + CHANNELS[ch][0]
        tex.location = (bsdf.location.x + x, bsdf.location.y - row * 280 + 300)
        row += 1
        if uvn is not None:
            nt.links.new(uvn.outputs[0], tex.inputs["Vector"])
        if ch == "NORMAL":
            nm = nt.nodes.new("ShaderNodeNormalMap")
            if uv_name:
                nm.uv_map = uv_name
            nm.location = (tex.location.x + 280, tex.location.y)
            nt.links.new(tex.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs[0], bsdf.inputs["Normal"])
        elif ch == "AO":
            continue
        else:
            sock = _shader_input(bsdf, CHANNELS[ch][2])
            if sock is not None:
                nt.links.new(tex.outputs["Color"], sock)
            if ch == "EMISSION":
                st = bsdf.inputs.get("Emission Strength")
                if st is not None:
                    st.default_value = 1.0
            if ch == "OPACITY":
                set_transparency_mode(mat)
    return True


def build_material(name, images, uv_name):
    old = bpy.data.materials.get(name)
    if old is not None:
        bpy.data.materials.remove(old)
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    if not any(n.bl_idname == "ShaderNodeBsdfPrincipled" for n in nt.nodes):
        out = output_node(nt) or nt.nodes.new("ShaderNodeOutputMaterial")
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(b.outputs[0], out.inputs["Surface"])
    wire_images(mat, images, uv_name)
    return mat


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

RES_ITEMS = [(str(r), str(r), "") for r in (512, 1024, 2048, 4096, 8192)]


class TK_MapsSettings(bpy.types.PropertyGroup):
    output_dir: StringProperty(name="Папка", subtype="DIR_PATH", default="//textures/")
    file_format: EnumProperty(name="Формат", items=[
        ("PNG", "PNG", ""), ("TARGA", "TGA", ""), ("TIFF", "TIFF", ""), ("JPEG", "JPEG", "")],
        default="PNG")

    ch_diffuse: BoolProperty(name="Diffuse", default=True)
    ch_opacity: BoolProperty(name="Opacity", default=True)
    ch_roughness: BoolProperty(name="Roughness", default=False)
    ch_metallic: BoolProperty(name="Metallic", default=False)
    ch_normal: BoolProperty(name="Normal", default=False)
    ch_emission: BoolProperty(name="Emission", default=False)
    ch_ao: BoolProperty(name="AO", default=False,
                        description="Ambient occlusion меша (только для атласа)")
    ao_samples: IntProperty(name="Сэмплов AO", default=32, min=1, max=4096)

    pack_rgba: BoolProperty(name="Diffuse + Opacity в одну RGBA", default=False,
                            description="Прозрачность в альфа-канале Diffuse (PNG/TGA/TIFF)")
    pack_orm: BoolProperty(name="ORM (AO, Roughness, Metallic)", default=False)
    keep_separate: BoolProperty(name="Сохранять и отдельные карты", default=True)

    resolution: EnumProperty(name="Разрешение", items=[("AUTO", "Как у текстур", "")] + RES_ITEMS,
                             default="AUTO")
    fallback_res: EnumProperty(name="Если текстур нет", items=RES_ITEMS, default="1024")

    bleed: BoolProperty(name="Растянуть цвет за край (edge padding)", default=True,
                        description="Убирает ореолы по краям листьев/вырезов при мипмаппинге")
    bleed_px: IntProperty(name="Пикселей", default=16, min=1, max=512)
    threshold: FloatProperty(name="Порог альфы", default=0.5, min=0.0, max=1.0)
    binary_mask: BoolProperty(name="Жёсткая маска (0/1)", default=False)
    invert: BoolProperty(name="Инвертировать Opacity", default=False)

    scope: EnumProperty(name="Материалы", items=[
        ("ACTIVE", "Активный материал", ""),
        ("SELECTED", "Выделенные объекты", ""),
        ("ALL", "Все материалы файла", "")], default="SELECTED")
    relink: BoolProperty(name="Подключить карты в материал", default=False)

    atlas_size: EnumProperty(name="Размер атласа", items=RES_ITEMS[:5], default="2048")
    atlas_mode: EnumProperty(name="Способ", items=[
        ("GRID", "Клетки по материалам", "Каждый материал — своя клетка, UV переносятся в неё "
                                          "(быстро; тайлинг UV за 0–1 не поддерживается)"),
        ("REPACK", "Перепаковать UV и запечь", "UV-острова пакуются в новую развёртку и "
                                               "запекаются с меша (поддерживает тайлинг и "
                                               "процедурные материалы)"),
    ], default="GRID")
    atlas_padding: IntProperty(name="Отступ, px", default=8, min=0, max=128)
    atlas_on_copy: BoolProperty(name="Работать на копии объекта", default=True)
    atlas_replace: BoolProperty(name="Заменить материалы одним атласным", default=True)

    def channels(self, allow_ao=True):
        out = []
        for ch in ORDER:
            if getattr(self, "ch_" + ch.lower()) and (allow_ao or ch != "AO"):
                out.append(ch)
        if self.pack_orm:
            for c in ("ROUGHNESS", "METALLIC") + (("AO",) if allow_ao else ()):
                if c not in out:
                    out.append(c)
        if self.pack_rgba:
            for c in ("DIFFUSE", "OPACITY"):
                if c not in out:
                    out.append(c)
        return [c for c in ORDER if c in out]


def check_out_dir(op, p):
    if p.output_dir.startswith("//") and not bpy.data.filepath:
        op.report({"ERROR"}, "Сохраните .blend или укажите абсолютный путь к папке")
        return None
    d = bpy.path.abspath(p.output_dir)
    os.makedirs(d, exist_ok=True)
    return d


def write_outputs(base, arrays, p, out_dir):
    """Save channel arrays and packed maps; returns dict ch -> Image (for wiring)."""
    ext = EXT[p.file_format]
    images = {}
    for ch, arr in arrays.items():
        if ch == "OPACITY" and arr[..., 0].min() >= 0.999 and not p.invert:
            continue
        name = base + CHANNELS[ch][1]
        path = os.path.join(out_dir, name + ext) if p.keep_separate or not (
            (p.pack_rgba and ch in ("DIFFUSE", "OPACITY")) or
            (p.pack_orm and ch in ("AO", "ROUGHNESS", "METALLIC"))) else None
        images[ch] = save_array(name, arr, path, p.file_format, CHANNELS[ch][3])
    for suffix, arr, srgb, alpha in packed_outputs(arrays, p):
        fmt = p.file_format
        if alpha and fmt == "JPEG":
            fmt = "PNG"
        name = base + suffix
        save_array(name, arr, os.path.join(out_dir, name + EXT[fmt]), fmt, srgb, alpha)
    return images


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def collect_materials(context, scope):
    if scope == "ACTIVE":
        ob = context.active_object
        return [ob.active_material] if ob and ob.active_material else []
    if scope == "SELECTED":
        mats = []
        for ob in context.selected_objects:
            for slot in getattr(ob, "material_slots", []):
                if slot.material and slot.material not in mats:
                    mats.append(slot.material)
        return mats
    return [m for m in bpy.data.materials if not m.is_grease_pencil]


def _report_warnings(op, warnings, what):
    if warnings:
        print(f"[TK {what}] Предупреждения:\n  " + "\n  ".join(warnings))
        op.report({"WARNING"}, f"Предупреждений: {len(warnings)} — первое: {warnings[0]} "
                               "(все — в системной консоли)")


class TK_OT_maps_extract(bpy.types.Operator):
    bl_idname = "tk.maps_extract"
    bl_label = "Извлечь карты"
    bl_description = ("Запечь выбранные каналы каждого материала в отдельные текстуры "
                      "(работает и с процедурными нодами)")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        p = context.scene.tk_maps
        out_dir = check_out_dir(self, p)
        if out_dir is None:
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        channels = p.channels(allow_ao=False)
        if not channels:
            self.report({"WARNING"}, "Не выбрано ни одного канала")
            return {"CANCELLED"}
        mats = collect_materials(context, p.scope)
        if not mats:
            self.report({"WARNING"}, "Нет материалов")
            return {"CANCELLED"}
        done, warnings = 0, []
        wm = context.window_manager
        wm.progress_begin(0, len(mats))
        try:
            for i, mat in enumerate(mats):
                wm.progress_update(i)
                miss = missing_images(mat)
                if miss:
                    warnings.append(f"{mat.name}: не загружены картинки {', '.join(miss)}")
                if p.resolution == "AUTO":
                    w, h = material_resolution(mat, int(p.fallback_res))
                else:
                    w = h = int(p.resolution)
                try:
                    arrays = bake_material_arrays(context, mat, channels, w, h)
                except RuntimeError as ex:
                    warnings.append(f"{mat.name}: {ex}")
                    continue
                for wmsg in postprocess(arrays, p):
                    warnings.append(f"{mat.name}: {wmsg}")
                base = bpy.path.clean_name(mat.name)
                images = write_outputs(base, arrays, p, out_dir)
                if p.relink:
                    if not wire_images(mat, images):
                        warnings.append(f"{mat.name}: нет Principled BSDF — карты не подключены")
                done += 1
        finally:
            wm.progress_end()
        self.report({"INFO"}, f"Готово: {done} материал(ов) → {out_dir}")
        _report_warnings(self, warnings, "Maps")
        return {"FINISHED"}


def _copy_object(context, ob):
    new = ob.copy()
    new.data = ob.data.copy()
    new.name = ob.name + "_Atlas"
    for coll in ob.users_collection:
        coll.objects.link(new)
    ob.hide_set(True)
    ob.select_set(False)
    return new


def used_material_tiles(ob):
    me = ob.data
    npoly = len(me.polygons)
    mat_idx = np.zeros(npoly, np.int32)
    if npoly:
        me.polygons.foreach_get("material_index", mat_idx)
    nslots = max(1, len(ob.material_slots))
    mat_idx = np.clip(mat_idx, 0, nslots - 1)
    keys, mats, slot_to_tile = [], [], {}
    for s in sorted(set(mat_idx.tolist())):
        m = ob.material_slots[s].material if s < len(ob.material_slots) else None
        key = m.name if m else f"__empty_{s}"
        if key not in keys:
            keys.append(key)
            mats.append(m)
        slot_to_tile[s] = keys.index(key)
    return mat_idx, mats, slot_to_tile


def grid_layout(n, A, pad):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    tw, th = A // cols, A // rows
    pad = min(pad, tw // 4, th // 4)
    return cols, rows, tw, th, pad


def remap_uvs_to_grid(me, src_uv_name, mat_idx, slot_to_tile, tiles, iw, ih, A, uv_name):
    npoly = len(me.polygons)
    nloops = len(me.loops)
    src = me.uv_layers[src_uv_name]
    uvs = np.empty(nloops * 2, np.float32)
    src.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    new_uv = me.uv_layers.get(uv_name)
    if new_uv is None:
        new_uv = me.uv_layers.new(name=uv_name)
        if new_uv is None:
            raise RuntimeError("превышен лимит UV-слоёв (8)")
    ls = np.empty(npoly, np.int32)
    lt = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    tile_of_poly = np.array([slot_to_tile[int(m)] for m in mat_idx], np.int32)
    loop_tile = np.repeat(tile_of_poly, lt)
    tx = np.array([t[0] for t in tiles], np.float32)
    ty = np.array([t[1] for t in tiles], np.float32)
    out = np.empty_like(uvs)
    out[:, 0] = (tx[loop_tile] + uvs[:, 0] * iw) / A
    out[:, 1] = (ty[loop_tile] + uvs[:, 1] * ih) / A
    new_uv.data.foreach_set("uv", out.ravel())
    return int(np.count_nonzero((uvs < -0.001) | (uvs > 1.001)))


def pack_uv_islands(context, ob, uv_name, margin):
    """Pack islands of *uv_name* into 0–1. Returns True on success."""
    me = ob.data
    prev_active = me.uv_layers.active.name if me.uv_layers.active else uv_name
    me.uv_layers.active = me.uv_layers[uv_name]
    vl = context.view_layer
    prev_obj = vl.objects.active
    vl.objects.active = ob
    ok = True
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action="SELECT")
        ts = context.tool_settings
        prev_sync = ts.use_uv_select_sync
        ts.use_uv_select_sync = True
        try:
            kw = dict(margin=margin, rotate=True)
            try:
                bpy.ops.uv.pack_islands(shape_method="CONCAVE", **kw)
            except TypeError:
                bpy.ops.uv.pack_islands(**kw)
        finally:
            ts.use_uv_select_sync = prev_sync
    except RuntimeError as ex:
        print("[TK Atlas] pack_islands:", ex)
        ok = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        me = ob.data
        if prev_active in me.uv_layers:
            me.uv_layers.active = me.uv_layers[prev_active]
        vl.objects.active = prev_obj
    return ok


class TK_OT_maps_atlas(bpy.types.Operator):
    bl_idname = "tk.maps_atlas"
    bl_label = "Собрать атлас"
    bl_description = ("Собрать все материалы меша в один набор текстур, перенести UV и "
                      "поставить один материал")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" for o in context.selected_objects)

    def execute(self, context):
        p = context.scene.tk_maps
        out_dir = check_out_dir(self, p)
        if out_dir is None:
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if not p.channels():
            self.report({"WARNING"}, "Не выбрано ни одного канала")
            return {"CANCELLED"}
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        results, warnings = [], []
        for ob in objs:
            try:
                r = self.process(context, ob, p, out_dir, warnings)
            except RuntimeError as ex:
                warnings.append(f"{ob.name}: {ex}")
                r = None
            if r:
                results.append(r)
        _report_warnings(self, warnings, "Atlas")
        if results:
            self.report({"INFO"}, "Атлас готов: " + "; ".join(results) + f" → {out_dir}")
            return {"FINISHED"}
        return {"CANCELLED"}

    def process(self, context, ob, p, out_dir, warnings):
        me = ob.data
        if not me.uv_layers:
            warnings.append(f"{ob.name}: нет UV-развёртки — пропущен")
            return None
        if not ob.material_slots:
            warnings.append(f"{ob.name}: нет материалов — пропущен")
            return None
        if p.atlas_on_copy:
            ob = _copy_object(context, ob)
            me = ob.data
            ob.select_set(True)
            context.view_layer.objects.active = ob
        src_uv = next((u for u in me.uv_layers if u.active_render), me.uv_layers.active)
        src_name = src_uv.name
        uv_name = "UV_Atlas"
        if src_name == uv_name:
            warnings.append(f"{ob.name}: исходная развёртка уже называется UV_Atlas — "
                            "она будет перезаписана")
        A = int(p.atlas_size)
        mat_idx, mats, slot_to_tile = used_material_tiles(ob)
        base = bpy.path.clean_name(ob.name.replace("_Atlas", "")) + "_Atlas"

        if p.atlas_mode == "GRID":
            arrays, info = self._grid(context, ob, p, A, mats, mat_idx, slot_to_tile,
                                      src_name, uv_name, warnings)
        else:
            arrays, info = self._repack(context, ob, p, A, src_name, uv_name, warnings)
        for wmsg in postprocess(arrays, p):
            warnings.append(f"{ob.name}: {wmsg}")
        images = write_outputs(base, arrays, p, out_dir)
        me.uv_layers.active = me.uv_layers[uv_name]
        if p.atlas_replace:
            amat = build_material(base, images, uv_name)
            me.materials.clear()
            me.materials.append(amat)
            me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int32))
            me.update()
        return f"{ob.name} ({len(mats)} мат., {info})"

    def _grid(self, context, ob, p, A, mats, mat_idx, slot_to_tile, src_name, uv_name, warnings):
        me = ob.data
        n = len(mats)
        cols, rows, tw, th, pad = grid_layout(n, A, p.atlas_padding)
        iw, ih = tw - 2 * pad, th - 2 * pad
        chans = p.channels(allow_ao=False)
        if p.ch_ao or p.pack_orm:
            warnings.append(f"{ob.name}: AO в режиме клеток не печётся — "
                            "используйте «Перепаковать UV и запечь»")
        arrays = {ch: np.zeros((A, A, 4), np.float32) for ch in chans}
        for ch in chans:
            arrays[ch][...] = (*DEFAULT_FILL[ch], 1.0)
        tiles = []
        for i, mat in enumerate(mats):
            c, r = i % cols, i // cols
            x0, y0 = c * tw, (rows - 1 - r) * th
            if mat is not None:
                for m in missing_images(mat):
                    warnings.append(f"{mat.name}: не загружена картинка {m}")
                tile = bake_material_arrays(context, mat, chans, iw, ih)
            else:
                tile = {ch: np.tile(np.array((*DEFAULT_FILL[ch], 1.0), np.float32), (ih, iw, 1))
                        for ch in chans}
            for ch in chans:
                t = tile[ch]
                if pad > 0:
                    t = np.pad(t, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                arrays[ch][y0:y0 + th, x0:x0 + tw] = t
            tiles.append((x0 + pad, y0 + pad))
        oor = remap_uvs_to_grid(me, src_name, mat_idx, slot_to_tile, tiles, iw, ih, A, uv_name)
        if oor:
            warnings.append(f"{ob.name}: {oor} UV-координат за пределами 0–1 (тайлинг) — "
                            "в клетках они залезут к соседям; выберите «Перепаковать UV и запечь»")
        return arrays, f"сетка {cols}×{rows}"

    def _repack(self, context, ob, p, A, src_name, uv_name, warnings):
        me = ob.data
        old = me.uv_layers.get(uv_name)
        if old is not None and uv_name != src_name:
            me.uv_layers.remove(old)
        src = me.uv_layers[src_name]
        new = me.uv_layers.new(name=uv_name)
        if new is None:
            raise RuntimeError("превышен лимит UV-слоёв (8)")
        buf = np.empty(len(me.loops) * 2, np.float32)
        src.data.foreach_get("uv", buf)
        new.data.foreach_set("uv", buf)
        for u in me.uv_layers:
            u.active_render = (u.name == src_name)
        if not pack_uv_islands(context, ob, uv_name, p.atlas_padding / A):
            raise RuntimeError("не удалось упаковать UV-острова")

        # empty slots need some material to bake
        tmp_mat = None
        empty = [i for i, s in enumerate(ob.material_slots) if s.material is None]
        if empty:
            tmp_mat = bpy.data.materials.new("TK_empty_slot")
            try:
                tmp_mat.use_nodes = True
            except Exception:
                pass
            for i in empty:
                ob.material_slots[i].material = tmp_mat
        arrays = {}
        try:
            for ch in p.channels():
                img = new_image(f"TK_tmp_{ch}", A, A, ch)
                try:
                    bake_channel(context, ob, ch, img, margin=max(2, p.atlas_padding),
                                 uv_layer=uv_name, ao_samples=p.ao_samples)
                    arrays[ch] = image_array(img).copy()
                finally:
                    bpy.data.images.remove(img)
        finally:
            if tmp_mat is not None:
                for i in empty:
                    ob.material_slots[i].material = None
                bpy.data.materials.remove(tmp_mat)
        for u in me.uv_layers:
            u.active_render = (u.name == uv_name)
        return arrays, "UV перепакованы"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def draw_maps(layout, context):
    p = context.scene.tk_maps
    layout.prop(p, "output_dir")
    layout.prop(p, "file_format")

    box = layout.box()
    box.label(text="Каналы", icon="NODE_TEXTURE")
    g = box.grid_flow(columns=2, align=True)
    for ch in ORDER:
        g.prop(p, "ch_" + ch.lower())
    if p.ch_ao:
        box.prop(p, "ao_samples")
    box.prop(p, "pack_rgba")
    box.prop(p, "pack_orm")
    if p.pack_rgba or p.pack_orm:
        box.prop(p, "keep_separate")

    box = layout.box()
    box.label(text="Прозрачность", icon="IMAGE_ALPHA")
    box.prop(p, "bleed")
    r = box.row()
    r.enabled = p.bleed
    r.prop(p, "bleed_px")
    box.prop(p, "threshold")
    box.prop(p, "binary_mask")
    box.prop(p, "invert")

    box = layout.box()
    box.label(text="1. Карты для каждого материала", icon="MATERIAL")
    box.prop(p, "scope")
    box.prop(p, "resolution")
    if p.resolution == "AUTO":
        box.prop(p, "fallback_res")
    box.prop(p, "relink")
    box.operator("tk.maps_extract", icon="IMAGE_RGB_ALPHA")

    box = layout.box()
    box.label(text="2. Атлас для выделенных мешей", icon="TEXTURE")
    box.prop(p, "atlas_mode", text="")
    box.prop(p, "atlas_size")
    box.prop(p, "atlas_padding")
    box.prop(p, "atlas_on_copy")
    box.prop(p, "atlas_replace")
    box.operator("tk.maps_atlas", icon="TEXTURE")


class TK_PT_maps(bpy.types.Panel):
    bl_label = "Карты и атлас"
    bl_idname = "TK_PT_maps"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context):
        draw_maps(self.layout, context)


class TK_PT_maps_node(bpy.types.Panel):
    bl_label = "Карты и атлас"
    bl_idname = "TK_PT_maps_node"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return getattr(context.space_data, "tree_type", "") == "ShaderNodeTree"

    def draw(self, context):
        draw_maps(self.layout, context)


classes = (TK_MapsSettings, TK_OT_maps_extract, TK_OT_maps_atlas, TK_PT_maps, TK_PT_maps_node)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_maps = PointerProperty(type=TK_MapsSettings)


def unregister():
    del bpy.types.Scene.tk_maps
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
