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
        layout.prop(s, "mode", expand=True)
        if ob is not None and ob.type == "MESH":
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
           TK_OT_split_pick_all, TK_UL_split_slots, TK_PT_split)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_split = PointerProperty(type=TK_SplitSettings)
    bpy.types.Material.tk_split_pick = BoolProperty(
        name="Вытащить", default=False, description="Вытащить этот материал в отдельный объект")


def unregister():
    del bpy.types.Material.tk_split_pick
    del bpy.types.Scene.tk_split
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
