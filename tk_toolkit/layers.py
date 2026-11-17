"""Texture layers: a Photoshop-like layer stack on a material.

Each layer is an image (plus optional mask) mixed on top of the previous ones with a
blend mode and opacity. The node tree is regenerated from the stack, so the material
always shows the live result, and the stack can be flattened into one image for export.
"""

import uuid

import bpy
import numpy as np
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)

from . import common as C

FLAG = "tk_layer_node"

CHANNELS = [
    ("BASE_COLOR", "Цвет", "Base Color", "COLOR"),
    ("ROUGHNESS", "Шероховатость", "Roughness", "MATFLUID"),
    ("METALLIC", "Металличность", "Metallic", "SHADING_RENDERED"),
]
CHANNEL_INPUT = {c[0]: c[2] for c in CHANNELS}
CHANNEL_ITEMS = [(c[0], c[1], "", c[3], i) for i, c in enumerate(CHANNELS)]

BLENDS = [
    ("MIX", "Обычный", ""),
    ("MULTIPLY", "Умножение", ""),
    ("SCREEN", "Экран", ""),
    ("OVERLAY", "Перекрытие", ""),
    ("ADD", "Добавление", ""),
    ("SUBTRACT", "Вычитание", ""),
    ("DARKEN", "Затемнение", ""),
    ("LIGHTEN", "Замена светлым", ""),
]

_building = {"on": False}


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

def _mat_of(layer):
    return layer.id_data


def _rebuild_cb(self, context):
    if not _building["on"]:
        rebuild(_mat_of(self))


def _opacity_cb(self, context):
    mat = _mat_of(self)
    nt = mat.node_tree
    node = nt.nodes.get(f"TK_op_{self.uid}") if nt else None
    if node is not None:
        node.inputs[1].default_value = self.opacity
    elif not _building["on"]:
        rebuild(mat)


def _select_cb(self, context):
    set_canvas(context, self)


class TK_Layer(bpy.types.PropertyGroup):
    name: StringProperty(name="Имя", default="Слой")
    uid: StringProperty()
    image: PointerProperty(name="Изображение", type=bpy.types.Image, update=_rebuild_cb)
    mask: PointerProperty(name="Маска", type=bpy.types.Image, update=_rebuild_cb)
    blend: EnumProperty(name="Режим", items=BLENDS, default="MIX", update=_rebuild_cb)
    opacity: FloatProperty(name="Непрозрачность", default=1.0, min=0.0, max=1.0,
                           subtype="FACTOR", update=_opacity_cb)
    visible: BoolProperty(name="Видимость", default=True, update=_rebuild_cb)
    channel: EnumProperty(name="Канал", items=CHANNEL_ITEMS, default="BASE_COLOR",
                          update=_rebuild_cb)
    edit_mask: BoolProperty(name="Рисовать маску", default=False,
                            description="Кисть рисует в маску слоя (белое — видно, чёрное — скрыто)",
                            update=lambda s, c: set_canvas(c, s))


# ---------------------------------------------------------------------------
# Node tree
# ---------------------------------------------------------------------------

def ensure_tree(mat):
    if mat.node_tree is None:
        try:
            mat.use_nodes = True
        except Exception:
            pass
    return mat.node_tree


def find_principled(nt):
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            return n
    out = None
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeOutputMaterial":
            out = n
            break
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.location = (out.location.x - 300, out.location.y)
    nt.links.new(b.outputs[0], out.inputs[0])
    return b


def _flag(n):
    n[FLAG] = 1
    return n


def rebuild(mat):
    """Regenerate the layer nodes of *mat* from its layer stack."""
    nt = ensure_tree(mat)
    if nt is None:
        return
    for n in [n for n in nt.nodes if n.get(FLAG)]:
        nt.nodes.remove(n)
    bsdf = find_principled(nt)
    layers_all = list(mat.tk_layers)
    if not layers_all:
        return
    bx, by = bsdf.location.x, bsdf.location.y
    frame = _flag(nt.nodes.new("NodeFrame"))
    frame.label = "Toolkit · слои (генерируется автоматически)"
    frame.name = "TK_frame"
    uv = _flag(nt.nodes.new("ShaderNodeUVMap"))
    uv.name = "TK_uv"
    uv.parent = frame

    row = 0
    col_x = bx - 1150
    for ch, _label, input_name, _icon in CHANNELS:
        sock = bsdf.inputs.get(input_name)
        if sock is None:
            continue
        # bottom → top
        layers = [l for l in reversed(layers_all) if l.channel == ch and l.visible and l.image]
        if not layers:
            continue
        base = _flag(nt.nodes.new("ShaderNodeRGB"))
        base.name = f"TK_base_{ch}"
        base.label = f"Основа: {input_name}"
        dv = sock.default_value
        try:
            base.outputs[0].default_value = (dv[0], dv[1], dv[2], 1.0)
        except TypeError:
            base.outputs[0].default_value = (dv, dv, dv, 1.0)
        base.location = (col_x + 250, by - row * 300 + 150)
        base.parent = frame
        current = base.outputs[0]
        for l in layers:
            y = by - row * 300
            tex = _flag(nt.nodes.new("ShaderNodeTexImage"))
            tex.name = f"TK_tex_{l.uid}"
            tex.label = l.name
            tex.image = l.image
            tex.location = (col_x, y)
            tex.parent = frame
            nt.links.new(uv.outputs[0], tex.inputs[0])

            op = _flag(nt.nodes.new("ShaderNodeMath"))
            op.name = f"TK_op_{l.uid}"
            op.label = "Непрозрачность"
            op.operation = "MULTIPLY"
            op.inputs[1].default_value = l.opacity
            op.location = (col_x + 300, y - 120)
            op.parent = frame
            nt.links.new(tex.outputs["Alpha"], op.inputs[0])
            fac = op.outputs[0]

            if l.mask:
                mt = _flag(nt.nodes.new("ShaderNodeTexImage"))
                mt.name = f"TK_mask_{l.uid}"
                mt.label = l.name + " · маска"
                mt.image = l.mask
                mt.location = (col_x, y - 270)
                mt.parent = frame
                nt.links.new(uv.outputs[0], mt.inputs[0])
                mm = _flag(nt.nodes.new("ShaderNodeMath"))
                mm.operation = "MULTIPLY"
                mm.location = (col_x + 300, y - 200)
                mm.parent = frame
                nt.links.new(fac, mm.inputs[0])
                nt.links.new(mt.outputs["Color"], mm.inputs[1])
                fac = mm.outputs[0]

            mix = _flag(nt.nodes.new("ShaderNodeMix"))
            mix.name = f"TK_mix_{l.uid}"
            mix.label = f"{l.name} · {dict((b[0], b[1]) for b in BLENDS)[l.blend]}"
            mix.data_type = "RGBA"
            mix.blend_type = l.blend
            mix.clamp_result = True
            mix.location = (col_x + 550, y)
            mix.parent = frame
            nt.links.new(fac, C.enabled_socket(mix.inputs, "Factor"))
            nt.links.new(current, C.enabled_socket(mix.inputs, "A"))
            nt.links.new(tex.outputs["Color"], C.enabled_socket(mix.inputs, "B"))
            current = C.enabled_socket(mix.outputs, "Result")
            row += 1
        if ch == "BASE_COLOR":
            nt.links.new(current, sock)
        else:
            bw = _flag(nt.nodes.new("ShaderNodeRGBToBW"))
            bw.name = f"TK_bw_{ch}"
            bw.location = (bx - 220, by - (row - 1) * 300)
            bw.parent = frame
            nt.links.new(current, bw.inputs[0])
            nt.links.new(bw.outputs[0], sock)
        row += 0.3


def set_canvas(context, layer):
    if layer is None or context is None:
        return
    img = layer.mask if (layer.edit_mask and layer.mask) else layer.image
    if img is None:
        return
    ts = context.scene.tool_settings
    try:
        ts.image_paint.mode = "IMAGE"
        ts.image_paint.canvas = img
    except Exception:
        pass
    mat = layer.id_data
    nt = mat.node_tree
    if nt:
        n = nt.nodes.get(f"TK_mask_{layer.uid}" if layer.edit_mask else f"TK_tex_{layer.uid}")
        if n:
            nt.nodes.active = n


def active_material(context):
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None
    return obj.active_material


# ---------------------------------------------------------------------------
# Flatten (numpy compositing, same formulas as Blender's Mix node)
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def blend(mode, A, B, f):
    f = f[..., None]
    if mode == "MIX":
        r = A * (1 - f) + B * f
    elif mode == "ADD":
        r = A + f * B
    elif mode == "SUBTRACT":
        r = A - f * B
    elif mode == "MULTIPLY":
        r = A * ((1 - f) + f * B)
    elif mode == "SCREEN":
        r = 1 - ((1 - f) + f * (1 - B)) * (1 - A)
    elif mode == "OVERLAY":
        low = A * ((1 - f) + 2 * f * B)
        high = 1 - ((1 - f) + 2 * f * (1 - B)) * (1 - A)
        r = np.where(A < 0.5, low, high)
    elif mode == "DARKEN":
        r = A * (1 - f) + np.minimum(A, B) * f
    elif mode == "LIGHTEN":
        r = A * (1 - f) + np.maximum(A, B) * f
    else:
        r = A * (1 - f) + B * f
    return np.clip(r, 0.0, 1.0)


def _is_srgb(img):
    try:
        return img.colorspace_settings.name.lower().startswith("srgb") and not img.is_float
    except Exception:
        return not img.is_float


def image_rgba(img, w, h):
    src = img
    tmp = None
    if tuple(img.size) != (w, h):
        tmp = img.copy()
        tmp.scale(w, h)
        src = tmp
    arr = np.empty(w * h * 4, np.float32)
    src.pixels.foreach_get(arr)
    if tmp is not None:
        bpy.data.images.remove(tmp)
    arr = arr.reshape(h, w, 4).astype(np.float64)
    if _is_srgb(img):
        arr[..., :3] = srgb_to_linear(arr[..., :3])
    return arr


def composite(mat, channel, w, h):
    """Return the flattened channel as linear RGB (h, w, 3)."""
    nt = ensure_tree(mat)
    bsdf = find_principled(nt)
    dv = bsdf.inputs[CHANNEL_INPUT[channel]].default_value
    try:
        base = np.array([dv[0], dv[1], dv[2]], np.float64)
    except TypeError:
        base = np.array([dv, dv, dv], np.float64)
    out = np.broadcast_to(base, (h, w, 3)).copy()
    for l in reversed(list(mat.tk_layers)):
        if l.channel != channel or not l.visible or not l.image:
            continue
        px = image_rgba(l.image, w, h)
        f = px[..., 3] * l.opacity
        if l.mask:
            m = image_rgba(l.mask, w, h)
            f = f * luminance(m[..., :3])  # implicit color→float in shader nodes
        out = blend(l.blend, out, px[..., :3], f)
    if channel != "BASE_COLOR":
        out = np.repeat(luminance(out)[..., None], 3, axis=-1)  # RGB to BW node
    return out


LUMA = np.array([0.2126, 0.7152, 0.0722])


def luminance(rgb):
    return rgb @ LUMA


def write_image(name, rgb, srgb):
    h, w = rgb.shape[:2]
    img = bpy.data.images.get(name)
    if img is None or tuple(img.size) != (w, h):
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, w, h, alpha=True)
    if not srgb:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    data = rgb.copy()
    if srgb:
        data = linear_to_srgb(data)
    rgba = np.concatenate([np.clip(data, 0, 1), np.ones((h, w, 1))], axis=-1)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    img.update()
    try:
        img.pack()
    except RuntimeError:
        pass
    return img


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

RES_ITEMS = [(str(r), str(r), "") for r in (512, 1024, 2048, 4096)]


class TK_OT_layer_add(bpy.types.Operator):
    bl_idname = "tk.layer_add"
    bl_label = "Добавить слой"
    bl_description = "Новый слой поверх текущих"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Имя", default="Слой")
    channel: EnumProperty(name="Канал", items=CHANNEL_ITEMS, default="BASE_COLOR")
    resolution: EnumProperty(name="Разрешение", items=RES_ITEMS, default="2048")
    fill: EnumProperty(name="Заливка", items=[
        ("TRANSPARENT", "Прозрачный", ""),
        ("COLOR", "Цвет", ""),
    ], default="TRANSPARENT")
    color: FloatVectorProperty(name="Цвет", subtype="COLOR", size=4, min=0, max=1,
                               default=(0.5, 0.5, 0.5, 1.0))
    blend: EnumProperty(name="Режим", items=BLENDS, default="MIX")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH"

    def invoke(self, context, event):
        mat = active_material(context)
        if mat is not None:
            self.name = f"Слой {len(mat.tk_layers) + 1}"
            self.channel = mat.tk_layers[mat.tk_layer_index].channel if len(mat.tk_layers) else "BASE_COLOR"
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        mat = obj.active_material
        if mat is None:
            mat = bpy.data.materials.new(obj.name + "_mat")
            ensure_tree(mat)
            obj.data.materials.append(mat)
            obj.active_material_index = len(obj.data.materials) - 1
        if not obj.data.uv_layers:
            self.report({"WARNING"}, "У меша нет UV-развёртки — слой не будет виден, сделайте Unwrap")
        res = int(self.resolution)
        img = bpy.data.images.new(f"{mat.name}_{self.name}", res, res, alpha=True)
        if self.fill == "TRANSPARENT":
            img.generated_color = (self.color[0], self.color[1], self.color[2], 0.0)
        else:
            img.generated_color = tuple(self.color)
        if self.channel != "BASE_COLOR":
            try:
                img.colorspace_settings.name = "Non-Color"
            except TypeError:
                pass
        _building["on"] = True
        try:
            l = mat.tk_layers.add()
            l.uid = uuid.uuid4().hex[:8]
            l.name = self.name
            l.channel = self.channel
            l.blend = self.blend
            l.image = img
            mat.tk_layers.move(len(mat.tk_layers) - 1, 0)
        finally:
            _building["on"] = False
        rebuild(mat)
        mat.tk_layer_index = 0
        return {"FINISHED"}


class TK_OT_layer_remove(bpy.types.Operator):
    bl_idname = "tk.layer_remove"
    bl_label = "Удалить слой"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        mat = active_material(context)
        return mat is not None and len(mat.tk_layers) > 0

    def execute(self, context):
        mat = active_material(context)
        i = mat.tk_layer_index
        mat.tk_layers.remove(i)
        mat.tk_layer_index = max(0, min(i, len(mat.tk_layers) - 1))
        rebuild(mat)
        return {"FINISHED"}


class TK_OT_layer_move(bpy.types.Operator):
    bl_idname = "tk.layer_move"
    bl_label = "Переместить слой"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=[("UP", "Вверх", ""), ("DOWN", "Вниз", "")])

    @classmethod
    def poll(cls, context):
        mat = active_material(context)
        return mat is not None and len(mat.tk_layers) > 1

    def execute(self, context):
        mat = active_material(context)
        i = mat.tk_layer_index
        j = i - 1 if self.direction == "UP" else i + 1
        if 0 <= j < len(mat.tk_layers):
            mat.tk_layers.move(i, j)
            mat.tk_layer_index = j
            rebuild(mat)
        return {"FINISHED"}


class TK_OT_layer_add_mask(bpy.types.Operator):
    bl_idname = "tk.layer_add_mask"
    bl_label = "Добавить маску"
    bl_description = "Белая маска (слой полностью виден); чёрным скрывается"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        mat = active_material(context)
        return mat is not None and len(mat.tk_layers) > 0

    def execute(self, context):
        mat = active_material(context)
        l = mat.tk_layers[mat.tk_layer_index]
        w, h = l.image.size if l.image else (2048, 2048)
        m = bpy.data.images.new(f"{mat.name}_{l.name}_mask", w, h, alpha=False)
        m.generated_color = (1, 1, 1, 1)
        try:
            m.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
        l.mask = m
        l.edit_mask = True
        return {"FINISHED"}


class TK_OT_layer_paint(bpy.types.Operator):
    bl_idname = "tk.layer_paint"
    bl_label = "Рисовать в слое"
    bl_description = "Перейти в Texture Paint и рисовать в выбранный слой (или его маску)"

    @classmethod
    def poll(cls, context):
        mat = active_material(context)
        return mat is not None and len(mat.tk_layers) > 0

    def execute(self, context):
        mat = active_material(context)
        l = mat.tk_layers[mat.tk_layer_index]
        if context.mode != "PAINT_TEXTURE":
            bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        set_canvas(context, l)
        for sp in C.view3d_spaces():
            if sp.shading.type == "SOLID":
                sp.shading.color_type = "TEXTURE"
        return {"FINISHED"}


class TK_OT_layers_flatten(bpy.types.Operator):
    bl_idname = "tk.layers_flatten"
    bl_label = "Свести слои"
    bl_description = "Собрать все видимые слои канала в одно изображение (для экспорта / движка)"
    bl_options = {"REGISTER", "UNDO"}

    channel: EnumProperty(name="Канал", items=CHANNEL_ITEMS, default="BASE_COLOR")
    replace: BoolProperty(name="Заменить слои результатом", default=False)

    @classmethod
    def poll(cls, context):
        mat = active_material(context)
        return mat is not None and len(mat.tk_layers) > 0

    def invoke(self, context, event):
        mat = active_material(context)
        self.channel = mat.tk_layers[mat.tk_layer_index].channel
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        mat = active_material(context)
        layers = [l for l in mat.tk_layers if l.channel == self.channel and l.image]
        if not layers:
            self.report({"WARNING"}, "В этом канале нет слоёв")
            return {"CANCELLED"}
        w = max(l.image.size[0] for l in layers)
        h = max(l.image.size[1] for l in layers)
        rgb = composite(mat, self.channel, w, h)
        img = write_image(f"{mat.name}_{self.channel.lower()}_flat", rgb,
                          srgb=self.channel == "BASE_COLOR")
        if self.replace:
            _building["on"] = True
            try:
                for i in reversed(range(len(mat.tk_layers))):
                    if mat.tk_layers[i].channel == self.channel:
                        mat.tk_layers.remove(i)
                l = mat.tk_layers.add()
                l.uid = uuid.uuid4().hex[:8]
                l.name = "Сведено"
                l.channel = self.channel
                l.image = img
                mat.tk_layers.move(len(mat.tk_layers) - 1, 0)
            finally:
                _building["on"] = False
            rebuild(mat)
            mat.tk_layer_index = 0
        self.report({"INFO"}, f"Готово: изображение «{img.name}» ({w}×{h}), упаковано в .blend")
        return {"FINISHED"}


class TK_OT_layers_rebuild(bpy.types.Operator):
    bl_idname = "tk.layers_rebuild"
    bl_label = "Пересобрать ноды"
    bl_description = "Пересоздать ноды слоёв, если их случайно изменили вручную"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return active_material(context) is not None

    def execute(self, context):
        rebuild(active_material(context))
        return {"FINISHED"}


def pack_layer_images(materials=None):
    n = 0
    for mat in (materials or bpy.data.materials):
        for l in getattr(mat, "tk_layers", []):
            for img in (l.image, l.mask):
                if img is None:
                    continue
                if img.packed_file is None or img.is_dirty:
                    try:
                        img.pack()
                        n += 1
                    except RuntimeError:
                        pass
    return n


@bpy.app.handlers.persistent
def _on_save_pre(*_a):
    pack_layer_images()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class TK_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "visible", text="", emboss=False,
                 icon="HIDE_OFF" if item.visible else "HIDE_ON")
        if item.image and item.image.preview:
            row.label(text="", icon_value=item.image.preview.icon_id)
        row.prop(item, "name", text="", emboss=False)
        ch_icon = {c[0]: c[3] for c in CHANNELS}[item.channel]
        row.label(text="", icon=ch_icon)
        if item.mask:
            row.prop(item, "edit_mask", text="", emboss=False,
                     icon="MOD_MASK" if item.edit_mask else "IMAGE_ALPHA")


class TK_PT_layers(bpy.types.Panel):
    bl_label = "Слои текстур"
    bl_idname = "TK_PT_layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            layout.label(text="Выберите меш", icon="INFO")
            return
        mat = obj.active_material
        if mat is None:
            layout.operator("tk.layer_add", text="Создать материал и слой", icon="ADD")
            return
        layout.label(text=f"Материал: {mat.name}", icon="MATERIAL")
        row = layout.row()
        row.template_list("TK_UL_layers", "", mat, "tk_layers", mat, "tk_layer_index", rows=5)
        col = row.column(align=True)
        col.operator("tk.layer_add", text="", icon="ADD")
        col.operator("tk.layer_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("tk.layer_move", text="", icon="TRIA_UP").direction = "UP"
        col.operator("tk.layer_move", text="", icon="TRIA_DOWN").direction = "DOWN"
        if len(mat.tk_layers):
            l = mat.tk_layers[mat.tk_layer_index]
            box = layout.box()
            box.prop(l, "blend")
            box.prop(l, "opacity", slider=True)
            box.prop(l, "channel")
            box.prop(l, "image")
            r = box.row(align=True)
            if l.mask:
                r.prop(l, "mask")
                box.prop(l, "edit_mask")
            else:
                r.operator("tk.layer_add_mask", icon="MOD_MASK")
            col = layout.column(align=True)
            col.scale_y = 1.3
            col.operator("tk.layer_paint", icon="BRUSH_DATA")
            row = layout.row(align=True)
            row.operator("tk.layers_flatten", icon="IMAGE_DATA")
            row.operator("tk.layers_rebuild", text="", icon="NODETREE")


classes = (
    TK_Layer,
    TK_OT_layer_add,
    TK_OT_layer_remove,
    TK_OT_layer_move,
    TK_OT_layer_add_mask,
    TK_OT_layer_paint,
    TK_OT_layers_flatten,
    TK_OT_layers_rebuild,
    TK_UL_layers,
    TK_PT_layers,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Material.tk_layers = CollectionProperty(type=TK_Layer)
    bpy.types.Material.tk_layer_index = IntProperty(
        default=0, update=lambda self, ctx: (len(self.tk_layers) and _select_cb(
            self.tk_layers[min(self.tk_layer_index, len(self.tk_layers) - 1)], ctx)) or None)
    bpy.app.handlers.save_pre.append(_on_save_pre)


def unregister():
    if _on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_on_save_pre)
    del bpy.types.Material.tk_layer_index
    del bpy.types.Material.tk_layers
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
