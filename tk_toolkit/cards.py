"""Geometry → card: bake a big mesh (fence, railing, grate…) onto a plane with
transparency, tile it seamlessly and replace the geometry with that plane.

Steps
  1. orientation: vertical card (Z up) along the main horizontal direction of the
     geometry (PCA of XY), facing the chosen side;
  2. period: the repeating step (fence span) is found by autocorrelation of an
     area-weighted 2D profile of the surface along the card (or set by hand);
  3. texel density: an integer number of spans must fill the width of the texture area
     exactly, with the density inside the range of the model type
       ВПМ — 512…1702 px/m, whole tile;
       НПМ — 10…40 px/m, one quarter of the tile chosen by the user;
  4. one segment (k spans) is baked selected-to-active onto the area of the atlas
     (Diffuse, Opacity, optionally Normal / Roughness / AO);
  5. the geometry is replaced by a plane over its whole length whose UVs repeat the
     segment (one quad with wrapping on a whole tile, a strip of quads in a quarter).
"""

import math
import os

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from mathutils import Vector

from . import common as C
from . import maps as M

HIGH_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}

TYPE_DEFAULTS = {"VPM": (512.0, 1702.0), "NPM": (10.0, 40.0)}
QUADRANTS = {  # (col, row) in halves, row 0 = bottom
    "LT": (0, 1), "RT": (1, 1), "LB": (0, 0), "RB": (1, 0),
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _type_changed(self, context):
    self.density_min, self.density_max = TYPE_DEFAULTS[self.model_type]


class TK_CardSettings(bpy.types.PropertyGroup):
    model_type: EnumProperty(name="Тип модели", items=[
        ("VPM", "ВПМ", "Высокополигональная / ближний план: 512–1702 px/м, весь тайл"),
        ("NPM", "НПМ", "Низкополигональная / дальний план: 10–40 px/м, четверть тайла"),
    ], default="VPM", update=_type_changed)
    tile_size: EnumProperty(name="Тайл", items=[("2048", "2048", ""), ("4096", "4096", "")],
                            default="2048")
    density_min: FloatProperty(name="Плотность от, px/м", default=512.0, min=0.1)
    density_max: FloatProperty(name="до, px/м", default=1702.0, min=0.1)
    quadrant: EnumProperty(name="Четверть", items=[
        ("LT", "Левая верхняя", ""), ("RT", "Правая верхняя", ""),
        ("LB", "Левая нижняя", ""), ("RB", "Правая нижняя", "")], default="LT")

    period_mode: EnumProperty(name="Пролёт", items=[
        ("AUTO", "Найти сам", "Повторяющийся шаг ищется по геометрии"),
        ("MANUAL", "Задать", ""),
        ("NONE", "Без повтора", "Запечь весь объект целиком, без тайлинга"),
    ], default="AUTO")
    period: FloatProperty(name="Пролёт", default=2.5, min=0.01, subtype="DISTANCE")
    side: EnumProperty(name="Сторона", items=[
        ("FRONT", "Спереди", ""), ("BACK", "Сзади", "")], default="FRONT")

    ch_normal: BoolProperty(name="Normal", default=True)
    ch_roughness: BoolProperty(name="Roughness", default=False)
    ch_ao: BoolProperty(name="AO", default=False)
    ao_samples: FloatProperty(name="Сэмплов AO", default=32, min=1, max=1024)
    bleed: BoolProperty(name="Растянуть цвет за край", default=True)
    bleed_px: FloatProperty(name="Пикселей", default=16, min=1, max=256)
    threshold: FloatProperty(name="Порог альфы", default=0.5, min=0.0, max=1.0)
    binary_mask: BoolProperty(name="Жёсткая маска (0/1)", default=False)

    atlas_name: StringProperty(name="Атлас", default="Cards",
                               description="Имя набора текстур; карточки с одним именем делят тайл")
    originals: EnumProperty(name="Исходная геометрия", items=[
        ("HIDE", "Скрыть", ""), ("DELETE", "Удалить", ""), ("KEEP", "Оставить", "")],
        default="HIDE")
    save_files: BoolProperty(name="Сохранить текстуры в папку", default=False)
    output_dir: StringProperty(name="Папка", subtype="DIR_PATH", default="//textures/")
    file_format: EnumProperty(name="Формат", items=[
        ("PNG", "PNG", ""), ("TARGA", "TGA", ""), ("TIFF", "TIFF", "")], default="PNG")

    def channels(self):
        out = ["DIFFUSE", "OPACITY"]
        if self.ch_normal:
            out.append("NORMAL")
        if self.ch_roughness:
            out.append("ROUGHNESS")
        if self.ch_ao:
            out.append("AO")
        return out


# ---------------------------------------------------------------------------
# Analysis (pure numpy)
# ---------------------------------------------------------------------------

def gather_points(objs, depsgraph, samples=200000, seed=3):
    """Area-weighted random surface points of all objects (world space)."""
    tris_all, areas = [], []
    for o in objs:
        A = C.mesh_arrays(o, depsgraph)
        co = A.world_co()
        if len(A.tris) == 0:
            continue
        t = co[A.tris]                                   # (n, 3, 3)
        a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        tris_all.append(t)
        areas.append(a)
    if not tris_all:
        return np.zeros((0, 3))
    T = np.concatenate(tris_all)
    A = np.concatenate(areas)
    if A.sum() <= 0:
        return T.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(T), size=samples, p=A / A.sum())
    r1, r2 = rng.random(samples), rng.random(samples)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    t = T[pick]
    return t[:, 0] + (t[:, 1] - t[:, 0]) * r1[:, None] + (t[:, 2] - t[:, 0]) * r2[:, None]


def card_frame(points, side="FRONT"):
    """Vertical frame: u along the main horizontal direction, z up, n = facing."""
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    cov = xy.T @ xy
    w, v = np.linalg.eigh(cov)
    d = v[:, np.argmax(w)]
    u = Vector((d[0], d[1], 0.0)).normalized()
    if abs(u.x) >= abs(u.y):
        if u.x < 0:
            u = -u
    elif u.y < 0:
        u = -u
    z = Vector((0, 0, 1))
    n = u.cross(z).normalized()           # quads (u0,z0)→(u1,z0)→(u1,z1) face +n
    if side == "BACK":
        u, n = -u, -n
    return u, z, n


def project(points, u, z, n):
    U = np.array(u)
    Z = np.array(z)
    N = np.array(n)
    return points @ U, points @ Z, points @ N


def find_period(pu, pz, length, min_lag=0.05):
    """Dominant repeat step along u. Returns (period or None, score 0..1)."""
    if length <= 0 or len(pu) < 100:
        return None, 0.0
    nb = int(min(8192, max(256, length / 0.005)))
    bins = nb
    zb = 24
    h, _, _ = np.histogram2d(pu, pz, bins=(bins, zb))
    h = h - h.mean(axis=0, keepdims=True)
    # autocorrelation along u (zero padded → linear, not circular)
    F = np.fft.rfft(h, n=2 * bins, axis=0)
    ac = np.fft.irfft(F * np.conj(F), axis=0)[:bins].sum(axis=1)
    overlap = (bins - np.arange(bins)) / bins          # unbiased estimate
    ac = ac / np.maximum(overlap, 1e-6)
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    step = length / bins
    lo = max(2, int(min_lag / step))
    hi = int(bins / 2)                                # need ≥ 2 repeats
    if hi <= lo + 2:
        return None, 0.0
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(seg[i] for i in peaks)
    if best < 0.4:
        return None, float(best)
    first = next(i for i in peaks if seg[i] >= 0.9 * best)
    k = lo + first
    # parabolic refinement of the peak
    if 1 <= k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        off = 0.0
    return (k + off) * step, float(seg[first])


def plan(length, height, period, tile, model_type, dmin, dmax, quadrant="LT"):
    """Work out texture area, density, segment length and repeats.

    Returns dict: area (x0, y0, w, h) in px, density, seg_len, spans, repeats, tiled,
    wrap (whole tile → UV wrap), used_h_px, warnings.
    """
    warns = []
    if model_type == "VPM":
        area = (0, 0, tile, tile)
    else:
        cx, cy = QUADRANTS[quadrant]
        half = tile // 2
        area = (cx * half, cy * half, half, half)
    aw, ah = area[2], area[3]
    dcap = dmax
    if height > 0:
        dcap = min(dcap, ah / height)
        if dcap < dmin:
            warns.append(f"высота {height:.2f} м не помещается в {ah} px при "
                         f"{dmin:g} px/м — плотность снижена до {dcap:.1f} px/м")
    tiled = False
    spans = 1
    if period and length / period >= 2.0:
        spans = max(1, math.ceil(aw / (period * dcap) - 1e-9))
        seg = spans * period
        if seg < length - 1e-6:
            tiled = True
            d = aw / seg
            if d < dmin - 1e-6:
                warns.append(f"один пролёт {period:.2f} м при {dmin:g} px/м не влезает в "
                             f"{aw} px — плотность {d:.1f} px/м")
        else:
            seg = length
    else:
        seg = length
    if not tiled:
        d = min(dcap, aw / seg) if seg > 0 else dcap
        if d < dmin - 1e-6:
            warns.append(f"объект {length:.2f} м без повтора не влезает в диапазон: "
                         f"плотность {d:.1f} px/м (нужно ≥ {dmin:g})")
    used_w = aw if tiled else max(1, min(aw, round(seg * d)))
    used_h = max(1, min(ah, math.ceil(height * d)))
    return {
        "area": area, "density": d, "seg_len": seg, "spans": spans if tiled else 0,
        "repeats": length / seg if seg > 0 else 1.0, "tiled": tiled,
        "wrap": tiled and model_type == "VPM", "used_w": used_w, "used_h": used_h,
        "warnings": warns,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def frame_point(u, z, n, a, b, c):
    return u * a + z * b + n * c


def make_quad_object(name, corners, uvs, collection, material=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(c) for c in corners], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for i, d in enumerate(uv.data):
        d.uv = uvs[i]
    if material is not None:
        me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def build_card_mesh(name, u, z, n, u0, u1, z0, z1, nc, s0, seg, uv_rect, wrap, collection,
                    material):
    """Card over [u0,u1]×[z0,z1]. Texture phase: U = (u - s0) / seg (mod 1).

    uv_rect = (x0, y0, x1, y1) in 0..1 for the used part of the texture area.
    wrap=True → one quad with U beyond 0..1 (repeat); else a strip split at segment
    boundaries so every quad stays inside the area (atlas-safe).
    """
    x0, y0, x1, y1 = uv_rect
    verts, faces, uvs = [], [], []

    def V(a, b):
        p = frame_point(u, z, n, a, b, nc)
        return (p.x, p.y, p.z)

    if wrap:
        cuts = [u0, u1]
    else:
        k0 = math.floor((u0 - s0) / seg)
        cuts = [u0]
        k = k0 + 1
        while s0 + k * seg < u1 - 1e-6:
            cuts.append(s0 + k * seg)
            k += 1
        cuts.append(u1)
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if wrap:
            ua, ub = (a - s0) / seg, (b - s0) / seg
            U0 = x0 + ua * (x1 - x0)
            U1 = x0 + ub * (x1 - x0)
        else:
            base = math.floor((a - s0) / seg + 1e-9)
            fa = (a - s0) / seg - base
            fb = (b - s0) / seg - base
            U0 = x0 + fa * (x1 - x0)
            U1 = x0 + fb * (x1 - x0)
        j = len(verts)
        verts += [V(a, z0), V(b, z0), V(b, z1), V(a, z1)]
        faces.append((j, j + 1, j + 2, j + 3))
        uvs += [(U0, y0), (U1, y0), (U1, y1), (U0, y1)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uvl = me.uv_layers.new(name="UVMap")
    for i, d in enumerate(uvl.data):
        d.uv = uvs[i]
    if not wrap:
        # weld the strip so it is one continuous surface (UVs stay split)
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(me)
        bm.free()
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


# ---------------------------------------------------------------------------
# Atlas images & material
# ---------------------------------------------------------------------------

def atlas_image(atlas, ch, tile):
    name = f"{atlas}{M.CHANNELS[ch][1]}"
    img = bpy.data.images.get(name)
    if img is not None and tuple(img.size) == (tile, tile):
        return img
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, tile, tile, alpha=(ch == "DIFFUSE"))
    fill = (0, 0, 0) if ch == "OPACITY" else M.DEFAULT_FILL[ch]
    img.generated_color = (*fill, 1.0)
    if not M.CHANNELS[ch][3]:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    return img


def card_material(atlas, images):
    name = f"{atlas}_card"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        try:
            mat.use_nodes = True
        except Exception:
            pass
        M.wire_images(mat, images)
        mat.use_backface_culling = False
        M.set_transparency_mode(mat)
    else:
        # images are reused by name, so an existing material already points at them;
        # make sure channels added later get connected too
        have = {n.image.name for n in mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeTexImage" and n.image}
        missing = {ch: im for ch, im in images.items() if im.name not in have}
        if missing:
            M.wire_images(mat, missing, x=-900)
    return mat


# ---------------------------------------------------------------------------
# Bake
# ---------------------------------------------------------------------------

def bake_segment(context, sources, plane, ch, tile, extrusion, max_dist, margin, ao_samples):
    """Bake *ch* from *sources* onto *plane* into a fresh tile×tile array."""
    img = bpy.data.images.new(f"TK_card_{ch}", tile, tile, alpha=True)
    if ch == "OPACITY":
        img.generated_color = (0, 0, 0, 1)          # misses stay 0 = transparent
    else:
        f = M.DEFAULT_FILL[ch]
        img.generated_color = (*f, 1.0)
    if not M.CHANNELS[ch][3]:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    pmat = plane.data.materials[0]
    tex = pmat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    pmat.node_tree.nodes.active = tex
    kind = M.CHANNELS[ch][4]
    mats = []
    for o in sources:
        for slot in getattr(o, "material_slots", []):
            if slot.material is not None and slot.material not in mats:
                mats.append(slot.material)
    undos = [M.ensure_nodes(m) for m in mats]
    patches = []
    try:
        if kind == "EMIT":
            patches = [M._MatPatch(m, ch, None) for m in mats]
        with M._SceneForBake(context, plane, int(ao_samples) if kind == "AO" else 1,
                             extra=sources):
            kw = dict(type=kind, use_selected_to_active=True, use_cage=False,
                      cage_extrusion=extrusion, max_ray_distance=max_dist,
                      margin=0 if ch == "OPACITY" else margin, use_clear=False,
                      target="IMAGE_TEXTURES")
            if kind == "NORMAL":
                kw["normal_space"] = "TANGENT"
            try:
                bpy.ops.object.bake(margin_type="EXTEND", **kw)
            except TypeError:
                bpy.ops.object.bake(**kw)
        arr = M.image_array(img).copy()
    finally:
        for p in patches:
            p.restore()
        for u_ in undos:
            u_()
        pmat.node_tree.nodes.remove(tex)
        bpy.data.images.remove(img)
    return arr


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _sources(context):
    return [o for o in context.selected_objects if o.type in HIGH_TYPES and o.visible_get()]


def analyse(context, s):
    objs = _sources(context)
    if not objs:
        raise RuntimeError("Выделите геометрию (забор, решётку…)")
    dg = context.evaluated_depsgraph_get()
    pts = gather_points(objs, dg)
    if len(pts) == 0:
        raise RuntimeError("У выделенных объектов нет граней")
    u, z, n = card_frame(pts, s.side)
    pu, pz, pn = project(pts, u, z, n)
    # exact extents from vertices, not samples
    allv = []
    for o in objs:
        A = C.mesh_arrays(o, dg)
        allv.append(A.world_co())
    V = np.concatenate(allv)
    vu, vz, vn = project(V, u, z, n)
    info = {
        "objs": objs, "u": u, "z": z, "n": n,
        "u0": float(vu.min()), "u1": float(vu.max()),
        "z0": float(vz.min()), "z1": float(vz.max()),
        "n0": float(vn.min()), "n1": float(vn.max()),
    }
    info["length"] = info["u1"] - info["u0"]
    info["height"] = info["z1"] - info["z0"]
    info["thickness"] = info["n1"] - info["n0"]
    score = 0.0
    if s.period_mode == "AUTO":
        period, score = find_period(pu - info["u0"], pz, info["length"])
    elif s.period_mode == "MANUAL":
        period = s.period
    else:
        period = None
    info["period"] = period
    info["score"] = score
    tile = int(s.tile_size)
    info["plan"] = plan(info["length"], info["height"], period, tile, s.model_type,
                        s.density_min, s.density_max, s.quadrant)
    return info


def describe(info):
    p = info["plan"]
    lines = [f"Длина {info['length']:.2f} м, высота {info['height']:.2f} м, "
             f"толщина {info['thickness']:.2f} м"]
    if info["period"]:
        lines.append(f"Пролёт {info['period']:.3f} м"
                     + (f" (уверенность {info['score']:.2f})" if info["score"] else ""))
    else:
        lines.append("Повтор не найден — карточка целиком")
    if p["tiled"]:
        lines.append(f"В текстуре {p['spans']} пролёт(а) = {p['seg_len']:.2f} м, "
                     f"повторов ×{p['repeats']:.2f}")
    lines.append(f"Плотность {p['density']:.1f} px/м, занято {p['used_w']}×{p['used_h']} px "
                 f"из {p['area'][2]}×{p['area'][3]}")
    return lines, p["warnings"]


_last = {"lines": [], "warnings": []}


class TK_OT_card_analyse(bpy.types.Operator):
    bl_idname = "tk.card_analyse"
    bl_label = "Проверить"
    bl_description = "Посчитать пролёт, плотность и тайлинг без запекания"

    def execute(self, context):
        try:
            info = analyse(context, context.scene.tk_card)
        except RuntimeError as ex:
            self.report({"WARNING"}, str(ex))
            return {"CANCELLED"}
        lines, warns = describe(info)
        _last.update(lines=lines, warnings=warns)
        self.report({"INFO"}, " · ".join(lines))
        for w in warns:
            self.report({"WARNING"}, w)
        return {"FINISHED"}


class TK_OT_card_make(bpy.types.Operator):
    bl_idname = "tk.card_make"
    bl_label = "Сделать карточку"
    bl_description = ("Запечь выделенную геометрию на плоскость с прозрачностью, затайлить по "
                      "пролёту и заменить геометрию этой плоскостью")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.tk_card
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            info = analyse(context, s)
        except RuntimeError as ex:
            self.report({"WARNING"}, str(ex))
            return {"CANCELLED"}
        lines, warns = describe(info)
        _last.update(lines=lines, warnings=warns)
        p = info["plan"]
        tile = int(s.tile_size)
        u, z, n = info["u"], info["z"], info["n"]
        objs = info["objs"]
        x0, y0, aw, ah = p["area"]
        used_w, used_h = p["used_w"], p["used_h"]
        seg = p["seg_len"]
        # segment to bake: start where a whole number of periods begins near the middle
        if p["tiled"]:
            per = info["period"]
            mid = info["u0"] + info["length"] / 2 - seg / 2
            s0 = info["u0"] + max(0, math.floor((mid - info["u0"]) / per)) * per
            s0 = min(s0, info["u1"] - seg)
        else:
            s0 = info["u0"]
        z0 = info["z0"]
        z1 = z0 + used_h / p["density"]               # whole pixels in height
        eps = max(1e-4, info["thickness"] * 0.02)
        nc = info["n1"] + eps                          # plane just in front of everything
        extrusion = eps
        max_dist = info["thickness"] + 3 * eps

        # bake plane covering exactly the used pixels of the area
        rx0, ry0 = x0 / tile, y0 / tile
        rx1, ry1 = (x0 + used_w) / tile, (y0 + used_h) / tile
        a1 = s0 + used_w / p["density"]
        corners = [frame_point(u, z, n, s0, z0, nc), frame_point(u, z, n, a1, z0, nc),
                   frame_point(u, z, n, a1, z1, nc), frame_point(u, z, n, s0, z1, nc)]
        pmat = bpy.data.materials.new("TK_card_bake")
        try:
            pmat.use_nodes = True
        except Exception:
            pass
        plane = make_quad_object("TK_card_bake", corners,
                                 [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)],
                                 context.scene.collection, pmat)
        arrays = {}
        try:
            margin = max(4, int(s.bleed_px))
            for ch in s.channels():
                arrays[ch] = bake_segment(context, objs, plane, ch, tile, extrusion, max_dist,
                                          margin, s.ao_samples)
        finally:
            me = plane.data
            bpy.data.objects.remove(plane)
            bpy.data.meshes.remove(me)
            bpy.data.materials.remove(pmat)

        # crop the used rectangle, post-process, write into the atlas
        sl = (slice(y0, y0 + used_h), slice(x0, x0 + used_w))
        crop = {ch: a[sl].copy() for ch, a in arrays.items()}
        op = crop["OPACITY"][..., 0]
        if s.bleed and "DIFFUSE" in crop:
            d = crop["DIFFUSE"]
            d[..., :3] = M.bleed_colors(d[..., :3], op >= s.threshold, int(s.bleed_px))
        if s.binary_mask:
            op = (op >= s.threshold).astype(np.float32)
        crop["OPACITY"][..., 0] = crop["OPACITY"][..., 1] = crop["OPACITY"][..., 2] = op
        if "DIFFUSE" in crop:
            crop["DIFFUSE"][..., 3] = op
        if op.max() < 0.01:
            self.report({"ERROR"}, "Ничего не запеклось — проверьте, что геометрия видна "
                                   "с выбранной стороны")
            return {"CANCELLED"}

        images = {}
        for ch, c in crop.items():
            img = atlas_image(s.atlas_name, ch, tile)
            full = M.image_array(img).copy()
            full[sl] = c
            if ch != "DIFFUSE":
                full[sl][..., 3] = 1.0
            M.set_image_array(img, full)
            try:
                img.pack()
            except RuntimeError:
                pass
            images[ch] = img
        if s.save_files:
            self._save(s, images)

        # the card itself
        mat = card_material(s.atlas_name, images)
        if p["wrap"]:
            uv_rect = (rx0, ry0, rx1, ry1)                   # whole tile: plain repeat
        else:
            inset = 0.5 / tile                               # keep sampling inside the area
            uv_rect = (rx0 + inset, ry0, rx1 - inset, ry1)
        coll = objs[0].users_collection[0] if objs[0].users_collection else context.scene.collection
        name = f"{objs[0].name}_card"
        card = build_card_mesh(name, u, z, n, info["u0"], info["u1"], z0,
                               z0 + info["height"], (info["n0"] + info["n1"]) / 2, s0, seg,
                               _scale_v(uv_rect, used_h, info["height"], p["density"]),
                               p["wrap"], coll, mat)
        card["tk_card_density"] = p["density"]
        for o in objs:
            if s.originals == "HIDE":
                o.hide_set(True)
                o.hide_render = True
            elif s.originals == "DELETE":
                bpy.data.objects.remove(o)
        C.deselect_all(context.view_layer)
        card.select_set(True)
        context.view_layer.objects.active = card
        self.report({"INFO"}, "Карточка готова · " + " · ".join(lines))
        for w in warns:
            self.report({"WARNING"}, w)
        return {"FINISHED"}

    def _save(self, s, images):
        if s.output_dir.startswith("//") and not bpy.data.filepath:
            self.report({"WARNING"}, "Сохраните .blend, чтобы писать текстуры по пути //")
            return
        d = bpy.path.abspath(s.output_dir)
        os.makedirs(d, exist_ok=True)
        for ch, img in images.items():
            path = os.path.join(d, img.name + M.EXT[s.file_format])
            img.filepath_raw = path
            img.file_format = s.file_format
            img.save()


def _scale_v(uv_rect, used_h, height, density):
    """Card spans the true height; the texture holds ceil(height·d) rows — map exactly."""
    x0, y0, x1, y1 = uv_rect
    true_h = height * density
    frac = true_h / used_h if used_h else 1.0
    return (x0, y0, x1, y0 + (y1 - y0) * min(1.0, frac))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class TK_PT_cards(bpy.types.Panel):
    bl_label = "Карточки из геометрии"
    bl_idname = "TK_PT_cards"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    def draw(self, context):
        s = context.scene.tk_card
        layout = self.layout
        layout.label(text="Выделите забор / решётку / ограду", icon="INFO")
        layout.prop(s, "model_type", expand=True)
        layout.prop(s, "tile_size", expand=True)
        r = layout.row(align=True)
        r.prop(s, "density_min")
        r.prop(s, "density_max")
        if s.model_type == "NPM":
            layout.prop(s, "quadrant")
        layout.prop(s, "period_mode")
        if s.period_mode == "MANUAL":
            layout.prop(s, "period")
        layout.prop(s, "side", expand=True)

        box = layout.box()
        box.label(text="Каналы: Diffuse + Opacity и…", icon="NODE_TEXTURE")
        g = box.row(align=True)
        g.prop(s, "ch_normal", toggle=True)
        g.prop(s, "ch_roughness", toggle=True)
        g.prop(s, "ch_ao", toggle=True)
        box.prop(s, "bleed")
        if s.bleed:
            box.prop(s, "bleed_px")
        box.prop(s, "threshold")
        box.prop(s, "binary_mask")

        layout.prop(s, "atlas_name")
        layout.prop(s, "originals")
        layout.prop(s, "save_files")
        if s.save_files:
            layout.prop(s, "output_dir")
            layout.prop(s, "file_format")

        r = layout.row(align=True)
        r.operator("tk.card_analyse", icon="VIEWZOOM")
        r.scale_y = 1.3
        r.operator("tk.card_make", icon="IMAGE_PLANE")
        if _last["lines"]:
            box = layout.box()
            for ln in _last["lines"]:
                box.label(text=ln)
            for w in _last["warnings"]:
                box.label(text=w, icon="ERROR")


classes = (TK_CardSettings, TK_OT_card_analyse, TK_OT_card_make, TK_PT_cards)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_card = bpy.props.PointerProperty(type=TK_CardSettings)


def unregister():
    del bpy.types.Scene.tk_card
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
