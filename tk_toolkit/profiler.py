"""Scene profiler: find what makes a scene heavy (viewport, undo, memory) and fix it."""

import bpy
import numpy as np
from bpy.props import (CollectionProperty, EnumProperty, FloatProperty, IntProperty,
                       StringProperty)

from . import common as C

KINDS = [
    ("GEOMETRY", "Геометрия", "", "MESH_DATA", 0),
    ("MODIFIER", "Модификатор", "", "MODIFIER", 1),
    ("IMAGE", "Текстура", "", "IMAGE_DATA", 2),
    ("DUPLICATE", "Дубликаты", "", "DUPLICATE", 3),
    ("PARTICLES", "Частицы", "", "PARTICLES", 4),
    ("SCENE", "Сцена", "", "SCENE_DATA", 5),
]
KIND_ICON = {k[0]: k[3] for k in KINDS}


class TK_ProfItem(bpy.types.PropertyGroup):
    kind: EnumProperty(items=KINDS)
    target: StringProperty()      # object / image / mesh name
    sub: StringProperty()         # modifier name, etc.
    info: StringProperty()
    advice: StringProperty()
    fix: StringProperty()         # id of an available quick fix
    score: FloatProperty()


def _fmt_count(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f} млн"
    if n >= 1e3:
        return f"{n / 1e3:.0f} тыс"
    return str(int(n))


def _fmt_mb(b):
    return f"{b / (1024 * 1024):.0f} МБ" if b >= 1024 * 1024 else f"{b / 1024:.0f} КБ"


def _tri_count(me):
    n = len(me.polygons)
    if not n:
        return 0
    tot = np.empty(n, np.int64)
    me.polygons.foreach_get("loop_total", tot)
    return int((tot - 2).clip(min=0).sum())


def _mesh_signature(me):
    nv = len(me.vertices)
    if nv == 0:
        return None
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    step = max(1, nv // 64)
    sample = np.round(co[::step], 4)
    return (nv, len(me.polygons), hash(sample.tobytes()))


def analyze(context):
    """Return a list of dicts describing hot spots, sorted by cost."""
    dg = context.evaluated_depsgraph_get()
    items = []
    total_tris = 0
    total_mesh_bytes = 0
    seen_meshes = set()
    sigs = {}

    for obj in context.view_layer.objects:
        if obj.type != "MESH":
            if obj.type in {"CURVE", "FONT", "SURFACE"} and obj.visible_get():
                ev = obj.evaluated_get(dg)
                try:
                    me = ev.to_mesh()
                    tris = _tri_count(me) if me else 0
                finally:
                    ev.to_mesh_clear()
                if tris > 200_000:
                    items.append(dict(kind="GEOMETRY", target=obj.name, score=tris,
                                      info=f"{_fmt_count(tris)} треугольников из кривой",
                                      advice="Снизьте Resolution U / Bevel Resolution кривой",
                                      fix=""))
            continue

        me = obj.data
        base_tris = _tri_count(me)
        if me.name not in seen_meshes:
            seen_meshes.add(me.name)
            total_mesh_bytes += len(me.vertices) * 40 + len(me.loops) * 24 + len(me.polygons) * 24
            if me.users == 1 and len(me.vertices) > 2000:
                sig = _mesh_signature(me)
                if sig:
                    sigs.setdefault(sig, []).append(obj.name)

        visible = obj.visible_get()
        ev = obj.evaluated_get(dg)
        ev_tris = _tri_count(ev.data) if visible else 0
        total_tris += ev_tris

        if ev_tris > 300_000:
            ratio = ev_tris / max(1, base_tris)
            adv = ("Модификаторы умножают геометрию в ×%.0f — снизьте уровни во вьюпорте" % ratio
                   if ratio > 3 else
                   "Тяжёлый меш: скрывайте его при работе с другими объектами или используйте Decimate для вьюпорта")
            items.append(dict(kind="GEOMETRY", target=obj.name, score=ev_tris,
                              info=f"{_fmt_count(ev_tris)} треугольников во вьюпорте "
                                   f"(база {_fmt_count(base_tris)})",
                              advice=adv, fix=""))

        for m in obj.modifiers:
            if not m.show_viewport:
                continue
            if m.type == "SUBSURF" and m.levels >= 2:
                mult = 4 ** m.levels
                items.append(dict(kind="MODIFIER", target=obj.name, sub=m.name,
                                  score=base_tris * mult * 0.5,
                                  info=f"Subdivision: уровень {m.levels} во вьюпорте (×{mult} граней)",
                                  advice="Поставьте 1 во вьюпорте, для рендера оставьте как есть",
                                  fix="SUBSURF_VIEWPORT"))
            elif m.type == "MULTIRES" and m.levels >= 3:
                items.append(dict(kind="MODIFIER", target=obj.name, sub=m.name,
                                  score=base_tris * 4 ** m.levels * 0.3,
                                  info=f"Multires: Viewport {m.levels}",
                                  advice="Снизьте Viewport уровень, когда не скульптите",
                                  fix="MULTIRES_VIEWPORT"))
            elif m.type == "BOOLEAN":
                solver = getattr(m, "solver", "")
                if solver == "EXACT":
                    items.append(dict(kind="MODIFIER", target=obj.name, sub=m.name,
                                      score=base_tris * 3 + 50_000,
                                      info="Boolean (Exact) пересчитывается при каждом изменении",
                                      advice="Переключите на Fast или выключите во вьюпорте на время работы",
                                      fix="DISABLE_VIEWPORT"))
            elif m.type == "REMESH" and getattr(m, "mode", "") == "VOXEL" and m.voxel_size < 0.01:
                items.append(dict(kind="MODIFIER", target=obj.name, sub=m.name,
                                  score=ev_tris + 100_000,
                                  info=f"Voxel Remesh с размером {C.fmt_len(m.voxel_size)}",
                                  advice="Увеличьте Voxel Size или примените модификатор",
                                  fix="DISABLE_VIEWPORT"))
            elif m.type == "ARRAY" and getattr(m, "count", 0) > 50:
                items.append(dict(kind="MODIFIER", target=obj.name, sub=m.name,
                                  score=base_tris * m.count,
                                  info=f"Array: {m.count} копий реальной геометрии",
                                  advice="Для повторов лучше инстансы (Geometry Nodes / Instance on Points)",
                                  fix=""))
            elif m.type == "BEVEL" and getattr(m, "segments", 1) > 6:
                items.append(dict(kind="MODIFIER", target=obj.name, sub=m.name,
                                  score=base_tris * m.segments,
                                  info=f"Bevel: {m.segments} сегментов",
                                  advice="Обычно хватает 2–4 сегментов + Shade Smooth",
                                  fix=""))
            elif m.type in {"CLOTH", "FLUID", "SOFT_BODY"}:
                items.append(dict(kind="MODIFIER", target=obj.name, sub=m.name,
                                  score=base_tris * 5 + 20_000,
                                  info=f"Симуляция {m.type} пересчитывается при смене кадра",
                                  advice="Запеките кэш симуляции",
                                  fix=""))

        for ps in obj.particle_systems:
            st = ps.settings
            cnt = st.count * (st.rendered_child_count if st.child_type != "NONE" else 1)
            vcnt = st.count * (st.child_nbr if st.child_type != "NONE" else 1) * st.display_percentage / 100
            if vcnt > 50_000:
                items.append(dict(kind="PARTICLES", target=obj.name, sub=ps.name, score=vcnt * 10,
                                  info=f"{ps.name}: {_fmt_count(vcnt)} частиц во вьюпорте "
                                       f"(в рендере {_fmt_count(cnt)})",
                                  advice="Снизьте Display Amount / Viewport children",
                                  fix=""))

    # duplicated single-user meshes → instancing candidates
    for sig, names in sigs.items():
        if len(names) > 1:
            nv = sig[0]
            items.append(dict(kind="DUPLICATE", target=names[0], sub="|".join(names),
                              score=nv * len(names) * 2,
                              info=f"{len(names)} одинаковых меша по {_fmt_count(nv)} вершин",
                              advice="Сделайте их связанными копиями (одни данные) — экономия памяти и undo",
                              fix="LINK_MESHES"))

    # images
    by_path = {}
    total_img = 0
    for img in bpy.data.images:
        if img.type not in {"IMAGE", "UV_TEST"} or not img.has_data:
            continue
        w, h = img.size
        bpp = img.channels * (4 if img.is_float else 1)
        size = w * h * bpp
        total_img += size
        if img.filepath:
            by_path.setdefault(bpy.path.abspath(img.filepath), []).append(img.name)
        if size > 64 * 1024 * 1024:
            items.append(dict(kind="IMAGE", target=img.name, score=size / 40,
                              info=f"{w}×{h}, {'float' if img.is_float else '8 бит'} — {_fmt_mb(size)}",
                              advice="Для вьюпорта хватит 2K–4K; float нужен только для HDR/дисплейсмента",
                              fix=""))
        if img.users == 0:
            items.append(dict(kind="IMAGE", target=img.name, score=size / 80,
                              info=f"Не используется, но загружена в память ({_fmt_mb(size)})",
                              advice="Удалите через File → Clean Up → Purge Unused Data",
                              fix=""))
    for path, names in by_path.items():
        if len(names) > 1:
            items.append(dict(kind="DUPLICATE", target=names[0], sub="|".join(names), score=1_000_000,
                              info=f"Один файл загружен {len(names)} раза: {bpy.path.basename(path)}",
                              advice="Переназначить все копии на одну",
                              fix="REMAP_IMAGES"))

    # scene summary / undo
    prefs = context.preferences.edit
    summary = {
        "tris": total_tris,
        "mesh_mb": total_mesh_bytes / (1024 * 1024),
        "img_mb": total_img / (1024 * 1024),
        "undo_steps": prefs.undo_steps,
        "undo_mem": prefs.undo_memory_limit,
    }
    if summary["mesh_mb"] > 300:
        items.append(dict(kind="SCENE", target="", score=summary["mesh_mb"] * 10_000,
                          info=f"Данные мешей ~{summary['mesh_mb']:.0f} МБ — "
                               "undo в Object Mode перечитывает изменённые блоки",
                          advice="Тяжёлые неизменяемые ассеты держите в отдельном .blend и подключайте Link",
                          fix=""))
    items.sort(key=lambda d: -d["score"])
    return items, summary


_summary = {}


class TK_OT_profile_scene(bpy.types.Operator):
    bl_idname = "tk.profile_scene"
    bl_label = "Проанализировать сцену"
    bl_description = "Найти, что тормозит вьюпорт, undo и съедает память"

    def execute(self, context):
        items, summary = analyze(context)
        coll = context.scene.tk_prof_items
        coll.clear()
        for d in items[:60]:
            it = coll.add()
            for k in ("kind", "target", "info", "advice", "fix"):
                setattr(it, k, d.get(k, ""))
            it.sub = d.get("sub", "")
            it.score = float(d["score"])
        _summary.clear()
        _summary.update(summary)
        self.report({"INFO"}, f"Найдено проблем: {len(items)}")
        return {"FINISHED"}


class TK_OT_profile_fix(bpy.types.Operator):
    bl_idname = "tk.profile_fix"
    bl_label = "Исправить"
    bl_description = "Применить предложенное исправление"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty()

    def execute(self, context):
        coll = context.scene.tk_prof_items
        if not (0 <= self.index < len(coll)):
            return {"CANCELLED"}
        it = coll[self.index]
        obj = bpy.data.objects.get(it.target)
        if it.fix == "SUBSURF_VIEWPORT" and obj:
            m = obj.modifiers.get(it.sub)
            if m:
                m.levels = min(m.levels, 1)
        elif it.fix == "MULTIRES_VIEWPORT" and obj:
            m = obj.modifiers.get(it.sub)
            if m:
                m.levels = min(m.levels, 1)
        elif it.fix == "DISABLE_VIEWPORT" and obj:
            m = obj.modifiers.get(it.sub)
            if m:
                m.show_viewport = False
        elif it.fix == "LINK_MESHES":
            names = it.sub.split("|")
            first = bpy.data.objects.get(names[0])
            if first:
                for n in names[1:]:
                    o = bpy.data.objects.get(n)
                    if o:
                        o.data = first.data
        elif it.fix == "REMAP_IMAGES":
            names = it.sub.split("|")
            keep = bpy.data.images.get(names[0])
            for n in names[1:]:
                img = bpy.data.images.get(n)
                if img and keep:
                    img.user_remap(keep)
        else:
            return {"CANCELLED"}
        coll.remove(self.index)
        return {"FINISHED"}


class TK_OT_profile_select(bpy.types.Operator):
    bl_idname = "tk.profile_select"
    bl_label = "Выделить"
    bl_description = "Выделить объект и показать его"

    index: IntProperty()

    def execute(self, context):
        it = context.scene.tk_prof_items[self.index]
        names = it.sub.split("|") if it.kind == "DUPLICATE" and "|" in it.sub else [it.target]
        objs = [bpy.data.objects.get(n) for n in names]
        objs = [o for o in objs if o and o.name in context.view_layer.objects]
        if not objs:
            self.report({"INFO"}, "Это не объект (текстура или сцена)")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        context.view_layer.objects.active = objs[0]
        try:
            bpy.ops.view3d.view_selected()
        except RuntimeError:
            pass
        return {"FINISHED"}


class TK_UL_prof(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        title = item.target + (f" · {item.sub}" if item.sub and item.kind != "DUPLICATE" else "")
        row.label(text=title or "Сцена", icon=KIND_ICON.get(item.kind, "DOT"))
        row.operator("tk.profile_select", text="", icon="RESTRICT_SELECT_OFF").index = index
        if item.fix:
            row.operator("tk.profile_fix", text="", icon="CHECKMARK").index = index


class TK_PT_profiler(bpy.types.Panel):
    bl_label = "Профайлер сцены"
    bl_idname = "TK_PT_profiler"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        layout.operator("tk.profile_scene", icon="VIEWZOOM")
        sc = context.scene
        if _summary:
            box = layout.box()
            box.label(text=f"Треугольников во вьюпорте: {_fmt_count(_summary['tris'])}")
            box.label(text=f"Меши ~{_summary['mesh_mb']:.0f} МБ · текстуры ~{_summary['img_mb']:.0f} МБ")
            box.label(text=f"Undo: {_summary['undo_steps']} шагов, лимит памяти "
                           f"{_summary['undo_mem'] or 'нет'} МБ")
        if len(sc.tk_prof_items):
            layout.template_list("TK_UL_prof", "", sc, "tk_prof_items", sc, "tk_prof_index", rows=6)
            i = sc.tk_prof_index
            if 0 <= i < len(sc.tk_prof_items):
                it = sc.tk_prof_items[i]
                box = layout.box()
                col = box.column(align=True)
                for chunk in _wrap(it.info, 42):
                    col.label(text=chunk)
                col.separator()
                for k, chunk in enumerate(_wrap(it.advice, 42)):
                    col.label(text=chunk, icon="LIGHT" if k == 0 else "BLANK1")
                if it.fix:
                    box.operator("tk.profile_fix", icon="CHECKMARK").index = i


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


classes = (TK_ProfItem, TK_OT_profile_scene, TK_OT_profile_fix, TK_OT_profile_select,
           TK_UL_prof, TK_PT_profiler)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_prof_items = CollectionProperty(type=TK_ProfItem)
    bpy.types.Scene.tk_prof_index = IntProperty()


def unregister():
    del bpy.types.Scene.tk_prof_index
    del bpy.types.Scene.tk_prof_items
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
