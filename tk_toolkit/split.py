"""Split a mesh by materials: all of them, or pull out only the chosen ones.

Unlike the built-in P → By Material, parts get readable names (Object_Material), keep
only their own material, can get their own origin, go into one collection, can be
exported one file per part, and can be joined back with the seams welded.
"""

import os

import bmesh
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty

from . import common as C

SRC_KEY = "tk_split_source"
NO_MAT = "БезМатериала"


class TK_SplitSettings(bpy.types.PropertyGroup):
    mode: EnumProperty(name="Что разделить", items=[
        ("ALL", "Все материалы", "Каждый материал — отдельный объект"),
        ("PICKED", "Только отмеченные", "Вытащить отмеченные материалы, остальное останется в объекте"),
        ("ACTIVE", "Только активный слот", "Вытащить материал активного слота"),
    ], default="ALL")
    extract: EnumProperty(name="Исходник", items=[
        ("CUT", "Вырезать", "Грани уходят из исходного объекта"),
        ("COPY", "Копировать", "Исходный объект остаётся целым"),
    ], default="CUT")
    on_copy: BoolProperty(name="Работать на копии (оригинал спрятать)", default=False,
                          description="Оригинал останется нетронутым и будет скрыт")
    origin: EnumProperty(name="Точка опоры", items=[
        ("KEEP", "Как у исходника", ""),
        ("BOUNDS", "Центр куска (габариты)", ""),
        ("MEDIAN", "Центр куска (среднее)", ""),
    ], default="BOUNDS")
    to_collection: BoolProperty(name="Сложить в коллекцию", default=True)

    export_after: BoolProperty(name="Экспортировать куски", default=False)
    export_format: EnumProperty(name="Формат", items=[
        ("FBX", "FBX", ""), ("OBJ", "OBJ", ""), ("GLB", "glTF (.glb)", "")], default="FBX")
    export_dir: StringProperty(name="Папка экспорта", subtype="DIR_PATH", default="//export/")
    export_reset_location: BoolProperty(
        name="Экспорт в начале координат", default=False,
        description="На время экспорта ставить кусок в (0,0,0)")

    highlight: BoolProperty(
        name="Подсветка материала", default=False,
        description="Подсвечивать во вьюпорте геометрию выбранного в списке материала "
                    "(и отмеченных галочками)",
        update=lambda self, ctx: C.tag_redraw_3d())
    weld_distance: FloatProperty(name="Сварка швов", default=1e-5, min=0.0, soft_max=0.01,
                                 subtype="DISTANCE", precision=6,
                                 description="При сборке сливать совпадающие вершины на швах")


def material_name(m):
    return m.name if m is not None else NO_MAT


def used_slots(ob):
    me = ob.data
    n = max(1, len(ob.material_slots))
    return sorted({min(max(p.material_index, 0), n - 1) for p in me.polygons})


def picked_slots(ob, s):
    if s.mode == "ALL":
        return used_slots(ob)
    if s.mode == "ACTIVE":
        return [ob.active_material_index]
    out = []
    for i, slot in enumerate(ob.material_slots):
        if slot.material is not None and slot.material.tk_split_pick:
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _select_only(context, objs, active=None):
    C.deselect_all(context.view_layer)
    for o in objs:
        o.select_set(True)
    context.view_layer.objects.active = active or (objs[0] if objs else None)


def keep_single_material(ob):
    """Leave only the material that the faces of *ob* use (first used slot)."""
    me = ob.data
    slots = used_slots(ob)
    mat = ob.material_slots[slots[0]].material if ob.material_slots and slots else None
    me.materials.clear()
    if mat is not None:
        me.materials.append(mat)
    for p in me.polygons:
        p.material_index = 0
    return mat


def _duplicate(context, ob, name):
    new = ob.copy()
    new.data = ob.data.copy()
    new.name = name
    for coll in ob.users_collection:
        coll.objects.link(new)
    return new


def split_object(context, ob, slots, extract="CUT"):
    """Separate the faces of material *slots* of *ob* into one object per material.

    Returns list of (part_object, material). With extract == 'CUT' the faces are
    removed from *ob*; with 'COPY' *ob* stays whole.
    """
    me = ob.data
    nslots = max(1, len(ob.material_slots))
    slot_set = set(slots)
    # group slots that hold the same material: they become one part
    groups = {}
    for i in slots:
        m = ob.material_slots[i].material if i < len(ob.material_slots) else None
        groups.setdefault(material_name(m), (m, set()))[1].add(i)

    parts = []
    all_used = set(used_slots(ob))
    picked_used = {i for i in slot_set if i in all_used}
    if not picked_used:
        return parts

    for key, (mat, idxs) in groups.items():
        if not idxs & all_used:
            continue
        part = _duplicate(context, ob, ob.name)
        bm = bmesh.new()
        bm.from_mesh(part.data)
        kill = [f for f in bm.faces if min(max(f.material_index, 0), nslots - 1) not in idxs]
        bmesh.ops.delete(bm, geom=kill, context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(part.data)
        bm.free()
        part.data.update()
        parts.append((part, mat))

    if extract == "CUT" and parts:
        bm = bmesh.new()
        bm.from_mesh(me)
        kill = [f for f in bm.faces if min(max(f.material_index, 0), nslots - 1) in picked_used]
        bmesh.ops.delete(bm, geom=kill, context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(me)
        bm.free()
        me.update()
    return parts


def set_origin(context, objs, how):
    if how == "KEEP" or not objs:
        return
    _select_only(context, objs)
    try:
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center=how)
    except RuntimeError:
        pass


def parts_collection(src_name, parent):
    name = f"{src_name}_по_материалам"
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        parent.children.link(coll)
    return coll


def move_to_collection(objs, coll):
    for o in objs:
        for c in list(o.users_collection):
            if c is not coll:
                c.objects.unlink(o)
        if coll not in o.users_collection:
            coll.objects.link(o)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

EXPORT_EXT = {"FBX": ".fbx", "OBJ": ".obj", "GLB": ".glb"}


def export_part(context, ob, fmt, path, reset_location=False):
    _select_only(context, [ob])
    loc = ob.location.copy()
    if reset_location:
        ob.location = (0, 0, 0)
    try:
        if fmt == "FBX":
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True)
        elif fmt == "OBJ":
            if hasattr(bpy.ops.wm, "obj_export"):
                bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True)
            else:
                bpy.ops.export_scene.obj(filepath=path, use_selection=True)
        else:
            bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB")
    finally:
        ob.location = loc


def export_parts(context, objs, s):
    if s.export_dir.startswith("//") and not bpy.data.filepath:
        raise RuntimeError("Сохраните .blend или укажите абсолютный путь для экспорта")
    d = bpy.path.abspath(s.export_dir)
    os.makedirs(d, exist_ok=True)
    paths = []
    for o in objs:
        path = os.path.join(d, bpy.path.clean_name(o.name) + EXPORT_EXT[s.export_format])
        export_part(context, o, s.export_format, path, s.export_reset_location)
        paths.append(path)
    return d, paths


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class TK_OT_split_by_material(bpy.types.Operator):
    bl_idname = "tk.split_by_material"
    bl_label = "Разделить по материалам"
    bl_description = ("Разделить меш на объекты по материалам (все или только отмеченные) "
                      "с понятными именами, своей точкой опоры и одним материалом в каждом")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH"

    def execute(self, context):
        s = context.scene.tk_split
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        src = context.active_object
        if len(src.material_slots) == 0:
            self.report({"WARNING"}, "У объекта нет материалов")
            return {"CANCELLED"}
        slots = picked_slots(src, s)
        if not slots:
            self.report({"WARNING"}, "Не отмечено ни одного материала")
            return {"CANCELLED"}
        if s.mode == "ALL" and len(used_slots(src)) < 2:
            self.report({"INFO"}, "Объект уже использует один материал")
        src_name = src.name
        parent = src.users_collection[0] if src.users_collection else context.scene.collection
        work = src
        extract = "CUT" if s.mode == "ALL" else s.extract
        if s.on_copy and extract == "CUT":
            work = _duplicate(context, src, src_name + "_части")
            src.hide_set(True)
            src.select_set(False)
        parts = split_object(context, work, slots, extract)
        if not parts:
            self.report({"WARNING"}, "Отмеченные материалы не используются гранями")
            return {"CANCELLED"}

        if extract == "CUT" and len(work.data.polygons) == 0:
            me = work.data
            bpy.data.objects.remove(work)
            if me.users == 0:
                bpy.data.meshes.remove(me)
        objs = []
        for part, mat in parts:
            keep_single_material(part)
            part.name = f"{src_name}_{material_name(mat)}"
            part[SRC_KEY] = src_name
            objs.append(part)
        set_origin(context, objs, s.origin)
        if s.to_collection:
            move_to_collection(objs, parts_collection(src_name, parent))
        _select_only(context, objs)

        msg = f"Частей: {len(objs)}"
        if s.export_after:
            try:
                d, paths = export_parts(context, objs, s)
                msg += f", экспортировано {len(paths)} → {d}"
            except RuntimeError as ex:
                self.report({"WARNING"}, str(ex))
            _select_only(context, objs)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class TK_OT_split_export(bpy.types.Operator):
    bl_idname = "tk.split_export"
    bl_label = "Экспортировать выделенные"
    bl_description = "Каждый выделенный объект — в отдельный файл"

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" for o in context.selected_objects)

    def execute(self, context):
        s = context.scene.tk_split
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        act = context.active_object
        try:
            d, paths = export_parts(context, objs, s)
        except RuntimeError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        _select_only(context, objs, act)
        self.report({"INFO"}, f"Экспортировано {len(paths)} → {d}")
        return {"FINISHED"}


def weld_seams(ob, dist):
    if dist <= 0:
        return 0
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    boundary = [v for v in bm.verts if v.is_boundary]
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=boundary, dist=dist)
    removed = before - len(bm.verts)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return removed


class TK_OT_split_join(bpy.types.Operator):
    bl_idname = "tk.split_join"
    bl_label = "Собрать обратно"
    bl_description = ("Объединить части обратно в один объект и сварить швы. Берутся выделенные "
                      "объекты; если выделена одна часть — все части того же исходника")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        s = context.scene.tk_split
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        act = context.active_object
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        src_name = act.get(SRC_KEY)
        if len(objs) < 2 and src_name:
            objs = [o for o in bpy.data.objects
                    if o.type == "MESH" and o.get(SRC_KEY) == src_name
                    and o.name in context.view_layer.objects]
        if len(objs) < 2:
            self.report({"WARNING"}, "Выделите хотя бы две части")
            return {"CANCELLED"}
        src_name = src_name or act.name
        target = act if act in objs else objs[0]
        colls = {c for o in objs for c in o.users_collection}
        _select_only(context, objs, target)
        with context.temp_override(active_object=target, object=target,
                                   selected_objects=objs, selected_editable_objects=objs):
            bpy.ops.object.join()
        target = context.view_layer.objects.active
        welded = weld_seams(target, s.weld_distance)
        if SRC_KEY in target:
            del target[SRC_KEY]
        target.name = src_name if bpy.data.objects.get(src_name) is None else src_name + "_собран"
        for c in colls:
            if c.name.endswith("_по_материалам") and len(c.objects) == len(
                    [o for o in c.objects if o is target]):
                parent = next((p for p in bpy.data.collections if c.name in p.children), None)
                dest = parent or context.scene.collection
                move_to_collection([target], dest)
                bpy.data.collections.remove(c)
        orig = bpy.data.objects.get(src_name)
        hint = ""
        if orig is not None and orig is not target and orig.hide_get():
            hint = f" (скрытый оригинал «{orig.name}» не тронут)"
        self.report({"INFO"}, f"Собрано: {target.name}, сварено вершин: {welded}{hint}")
        return {"FINISHED"}


import re as _re


def _img_key(img):
    if img is None:
        return None
    fp = bpy.path.abspath(img.filepath) if img.filepath else ""
    return fp or _re.sub(r"\.\d{3}$", "", img.name)


def _val(v, tol):
    """default_value → rounded, hashable."""
    try:
        seq = list(v)
    except TypeError:
        seq = None
    if seq is None:
        if isinstance(v, (int, float, bool)):
            return round(float(v) / tol)
        return str(v)
    return tuple(round(float(x) / tol) if isinstance(x, (int, float)) else str(x) for x in seq)


def material_signature(m, tol=0.01):
    """What a material IS (nodes, values, textures, links), not what it is called: two
    copies 'Metal', 'Metal.001' from an import get the same signature."""
    if m is None:
        return ("NONE",)
    if not m.use_nodes or m.node_tree is None:
        return ("FLAT", _val(m.diffuse_color, tol), _val(m.metallic, tol),
                _val(m.roughness, tol))
    nodes = []
    for n in m.node_tree.nodes:
        if n.bl_idname in ("NodeFrame", "NodeReroute"):
            continue
        ins = tuple((i.identifier, _val(i.default_value, tol)) for i in n.inputs
                    if not i.is_linked and hasattr(i, "default_value"))
        extra = []
        if getattr(n, "image", None) is not None:
            extra.append(("img", _img_key(n.image)))
        for attr in ("blend_type", "operation", "data_type", "interpolation", "projection",
                     "extension", "space", "uv_map", "distribution", "subsurface_method"):
            if hasattr(n, attr):
                extra.append((attr, str(getattr(n, attr))))
        nodes.append((n.bl_idname, ins, tuple(extra)))
    links = sorted((l.from_node.bl_idname, l.from_socket.identifier, l.to_node.bl_idname,
                    l.to_socket.identifier) for l in m.node_tree.links)
    return ("NODES", tuple(sorted(nodes, key=repr)), tuple(links),
            m.blend_method if hasattr(m, "blend_method") else "")


def merge_duplicate_slots(ob, tol=0.01):
    """Collapse material slots with identical content into one slot each. Returns
    (n_before, n_after, {kept_slot: [names merged into it]})."""
    me = ob.data
    mats = list(me.materials)
    if len(mats) < 2:
        return len(mats), len(mats), {}
    sig = [material_signature(m, tol) for m in mats]
    first, new_mats, remap, groups = {}, [], [], {}
    for i, sgn in enumerate(sig):
        if sgn not in first:
            first[sgn] = len(new_mats)
            new_mats.append(mats[i])
        k = first[sgn]
        remap.append(k)
        groups.setdefault(k, []).append(mats[i].name if mats[i] else "—")
    if len(new_mats) == len(mats):
        return len(mats), len(mats), {}
    mi = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", mi)
    mi = np.array(remap, np.int32)[np.clip(mi, 0, len(remap) - 1)]
    me.materials.clear()
    for m in new_mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", mi)
    me.update()
    return len(mats), len(new_mats), {k: v for k, v in groups.items() if len(v) > 1}


class TK_OT_split_merge_same(bpy.types.Operator):
    bl_idname = "tk.split_merge_same"
    bl_label = "Объединить одинаковые материалы"
    bl_description = ("Материалы с одинаковым содержимым (цвет, металл, шероховатость, "
                      "текстуры, ноды) сводятся в один слот — даже если называются по-разному. "
                      "Потом их можно вытащить одним куском")
    bl_options = {"REGISTER", "UNDO"}

    tolerance: bpy.props.FloatProperty(name="Допуск значений", default=0.01, min=0.0001,
                                       max=0.5, description="Значения, отличающиеся меньше "
                                       "допуска, считаются одинаковыми")
    extract: bpy.props.BoolProperty(name="Сразу вытащить самый большой набор", default=False,
                                    description="После объединения отделить в свой объект "
                                                "материал, в который слилось больше всего копий")

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" for o in context.selected_objects)

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        tot_b = tot_a = 0
        best = None
        for ob in [o for o in context.selected_objects if o.type == "MESH"]:
            b, a, groups = merge_duplicate_slots(ob, self.tolerance)
            tot_b += b
            tot_a += a
            for k, names in groups.items():
                if best is None or len(names) > best[2]:
                    best = (ob, k, len(names), names)
        if tot_b == tot_a:
            self.report({"INFO"}, "Одинаковых материалов не нашлось")
            return {"FINISHED"}
        msg = f"Слотов было {tot_b}, стало {tot_a}"
        if best is not None:
            ob, k, n, names = best
            context.view_layer.objects.active = ob
            ob.active_material_index = k
            msg += f"; больше всего слилось в «{names[0]}» ({n} шт.) — он выбран в списке"
            if self.extract:
                s = context.scene.tk_split
                old = s.mode
                s.mode = "ACTIVE"
                for o in context.selected_objects:
                    o.select_set(False)
                ob.select_set(True)
                try:
                    bpy.ops.tk.split_by_material()
                finally:
                    s.mode = old
                msg += " и вытащен в отдельный объект"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class TK_OT_split_pick_all(bpy.types.Operator):
    bl_idname = "tk.split_pick_all"
    bl_label = "Отметить все"
    bl_options = {"REGISTER", "UNDO"}

    value: BoolProperty(default=True)

    def execute(self, context):
        ob = context.active_object
        for slot in getattr(ob, "material_slots", []):
            if slot.material is not None:
                slot.material.tk_split_pick = self.value
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class TK_UL_split_slots(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        m = item.material
        if m is None:
            row.label(text="(пустой слот)", icon="BLANK1")
            return
        if context.scene.tk_split.mode == "PICKED":
            row.prop(m, "tk_split_pick", text="")
        row.label(text=m.name, icon_value=layout.icon(m))


class TK_PT_split(bpy.types.Panel):
    bl_label = "Разделение по материалам"
    bl_idname = "TK_PT_split"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context):
        s = context.scene.tk_split
        layout = self.layout
        ob = context.active_object
        layout.operator("tk.split_merge_same", icon="MATERIAL")
        layout.prop(s, "mode", expand=True)
        if ob is not None and ob.type == "MESH":
            layout.prop(s, "highlight", icon="LIGHT_SUN")
            layout.template_list("TK_UL_split_slots", "", ob, "material_slots", ob,
                                 "active_material_index", rows=4)
            if s.mode == "PICKED":
                r = layout.row(align=True)
                r.operator("tk.split_pick_all", text="Все").value = True
                r.operator("tk.split_pick_all", text="Ни одного").value = False
        if s.mode != "ALL":
            layout.prop(s, "extract", expand=True)
        if s.mode == "ALL" or s.extract == "CUT":
            layout.prop(s, "on_copy")
        layout.prop(s, "origin")
        layout.prop(s, "to_collection")
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("tk.split_by_material", icon="MOD_EXPLODE")
        r = layout.row(align=True)
        r.operator("tk.split_join", icon="AUTOMERGE_ON")
        r.prop(s, "weld_distance", text="")

        box = layout.box()
        box.label(text="Экспорт кусков", icon="EXPORT")
        box.prop(s, "export_format")
        box.prop(s, "export_dir")
        box.prop(s, "export_reset_location")
        box.prop(s, "export_after")
        box.operator("tk.split_export", icon="EXPORT")


classes = (TK_SplitSettings, TK_OT_split_by_material, TK_OT_split_export, TK_OT_split_join,
           TK_OT_split_merge_same, TK_OT_split_pick_all, TK_UL_split_slots, TK_PT_split)


# ---------------------------------------------------------------------------
# Highlight of the chosen material in the viewport
# ---------------------------------------------------------------------------

import numpy as np

COL_ACTIVE = (0.2, 0.85, 1.0)     # material selected in the list
COL_PICKED = (1.0, 0.55, 0.1)     # materials ticked for extraction

_hl = {"handle": None, "key": None, "batches": [], "geo": {}}


def highlight_sets(ob, s):
    """[(color, set_of_slot_indices)] to draw for *ob*."""
    out = []
    picked = set()
    if s.mode == "PICKED":
        picked = {i for i, sl in enumerate(ob.material_slots)
                  if sl.material is not None and sl.material.tk_split_pick}
    act = ob.active_material_index
    if picked:
        out.append((COL_PICKED, picked - {act}))
    if 0 <= act < len(ob.material_slots):
        out.append((COL_ACTIVE, {act}))
    return [(c, sl) for c, sl in out if sl]


def material_region(ob, depsgraph, slots):
    """World-space triangles and outline segments of the faces using *slots*."""
    ev = ob.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        npoly = len(me.polygons)
        if npoly == 0:
            return None, None
        mi = np.empty(npoly, np.int32)
        me.polygons.foreach_get("material_index", mi)
        nsl = max(1, len(ob.material_slots))
        mi = np.clip(mi, 0, nsl - 1)
        sel = np.isin(mi, list(slots))
        if not sel.any():
            return None, None
        nv = len(me.vertices)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        try:
            vn = np.empty(nv * 3, np.float32)
            me.vertex_normals.foreach_get("vector", vn)
        except AttributeError:
            me.vertices.foreach_get("normal", vn)
        vn = vn.reshape(-1, 3)
        nt = len(me.loop_triangles)
        tris = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)
        tpoly = np.empty(nt, np.int64)
        me.loop_triangles.foreach_get("polygon_index", tpoly)
        tris = tris[sel[tpoly]]
        # boundary of the region: edges used by exactly one selected polygon
        ls = np.empty(npoly, np.int64)
        lt = np.empty(npoly, np.int64)
        me.polygons.foreach_get("loop_start", ls)
        me.polygons.foreach_get("loop_total", lt)
        lv = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get("vertex_index", lv)
        pol = np.repeat(np.arange(npoly), lt)
        li = np.arange(len(lv))
        nxt = ls[pol] + (li - ls[pol] + 1) % lt[pol]
        m = sel[pol]
        a, b = lv[li[m]], lv[nxt[m]]
        e = np.sort(np.stack([a, b], 1), axis=1)
        uniq, cnt = np.unique(e, axis=0, return_counts=True)
        edges = uniq[cnt == 1]
        mw = np.array([list(r) for r in ob.matrix_world], np.float32)
        wco = co @ mw[:3, :3].T + mw[:3, 3]
        n3 = np.linalg.inv(mw[:3, :3]).T
        wn = vn @ n3.T
        wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-9)
        diag = float(np.linalg.norm(wco.max(0) - wco.min(0))) or 1.0
        push = wco + wn * diag * 2e-4            # lift over the surface: no z-fighting
        return push[tris.reshape(-1)], push[edges.reshape(-1)]
    finally:
        ev.to_mesh_clear()


def _hl_key(ob, s):
    return (ob.name, s.mode, tuple(sorted((tuple(c), tuple(sorted(sl)))
                                          for c, sl in highlight_sets(ob, s))),
            _hl["geo"].get(ob.name, 0), tuple(round(x, 6) for r in ob.matrix_world for x in r))


def _hl_draw():
    ctx = bpy.context
    try:
        s = ctx.scene.tk_split
    except AttributeError:
        return
    ob = ctx.active_object
    if not s.highlight or ob is None or ob.type != "MESH" or ctx.mode != "OBJECT" \
            or not ob.visible_get() or len(ob.material_slots) == 0:
        return
    key = _hl_key(ob, s)
    if key != _hl["key"]:
        _hl["key"] = key
        _hl["batches"] = []
        dg = ctx.evaluated_depsgraph_get()
        for col, slots in highlight_sets(ob, s):
            tri, seg = material_region(ob, dg, slots)
            if tri is None:
                continue
            _hl["batches"].append((col, C.make_batch("uniform", "TRIS", tri),
                                   C.make_batch("uniform", "LINES", seg)))
    if not _hl["batches"]:
        return
    with C.GPUState(depth=False):                 # outline seen through everything
        for col, _f, edges in _hl["batches"]:
            C.draw_batch(edges, (*col, 0.35), 1.5)
    with C.GPUState(depth=True):
        for col, fill, edges in _hl["batches"]:
            C.draw_batch(fill, (*col, 0.28))
            C.draw_batch(edges, (*col, 1.0), 2.5)


@bpy.app.handlers.persistent
def _hl_depsgraph(scene, depsgraph):
    for u in depsgraph.updates:
        if isinstance(u.id, bpy.types.Object) and u.is_updated_geometry:
            name = u.id.original.name if hasattr(u.id, "original") else u.id.name
            _hl["geo"][name] = _hl["geo"].get(name, 0) + 1


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_split = PointerProperty(type=TK_SplitSettings)
    bpy.types.Material.tk_split_pick = BoolProperty(
        name="Вытащить", default=False, description="Вытащить этот материал в отдельный объект",
        update=lambda self, ctx: C.tag_redraw_3d())
    bpy.app.handlers.depsgraph_update_post.append(_hl_depsgraph)
    if C.can_draw():
        _hl["handle"] = bpy.types.SpaceView3D.draw_handler_add(_hl_draw, (), "WINDOW", "POST_VIEW")


def unregister():
    if _hl["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_hl["handle"], "WINDOW")
        _hl["handle"] = None
    if _hl_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_hl_depsgraph)
    _hl["batches"] = []
    _hl["key"] = None
    del bpy.types.Material.tk_split_pick
    del bpy.types.Scene.tk_split
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
