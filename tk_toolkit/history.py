"""Readable action history: what you did, in plain words, and what Ctrl+Z will undo next.

Blender does not expose undo-step names to Python, so the log is built from the
operators that create undo steps (`window_manager.operators` with the UNDO flag), and
kept in sync with Ctrl+Z / Ctrl+Shift+Z through the undo/redo handlers. Steps that
Blender records without a registered operator can be missing from the list.
"""

import math
import time

import bpy
from bpy.props import BoolProperty, IntProperty

from . import common as C

MAX_LOG = 150

_log = []           # list of dicts
_ptr = {"i": -1, "last": 0, "h2d": None}

SKIP = {
    "ED_OT_undo", "ED_OT_redo", "ED_OT_undo_history", "ED_OT_undo_push",
    "TK_OT_history_goto", "SCREEN_OT_repeat_last", "SCREEN_OT_redo_last",
}

NAMES = {
    "TRANSFORM_OT_translate": "Перемещение",
    "TRANSFORM_OT_rotate": "Поворот",
    "TRANSFORM_OT_resize": "Масштаб",
    "TRANSFORM_OT_shrink_fatten": "Сжатие / раздувание",
    "TRANSFORM_OT_edge_slide": "Скольжение ребра",
    "TRANSFORM_OT_vert_slide": "Скольжение вершины",
    "TRANSFORM_OT_tosphere": "К сфере",
    "TRANSFORM_OT_shear": "Скос",
    "TRANSFORM_OT_mirror": "Отражение",
    "MESH_OT_extrude_region_move": "Экструзия",
    "MESH_OT_extrude_region_shrink_fatten": "Экструзия по нормалям",
    "MESH_OT_extrude_context_move": "Экструзия",
    "MESH_OT_extrude_faces_move": "Экструзия граней",
    "MESH_OT_extrude_vertices_move": "Экструзия вершин",
    "MESH_OT_extrude_edges_move": "Экструзия рёбер",
    "MESH_OT_loopcut_slide": "Петлевой разрез",
    "MESH_OT_bevel": "Фаска",
    "MESH_OT_inset": "Вставка граней",
    "MESH_OT_delete": "Удаление",
    "MESH_OT_dissolve_edges": "Растворение рёбер",
    "MESH_OT_dissolve_verts": "Растворение вершин",
    "MESH_OT_dissolve_faces": "Растворение граней",
    "MESH_OT_knife_tool": "Нож",
    "MESH_OT_subdivide": "Подразделение",
    "MESH_OT_merge": "Слияние вершин",
    "MESH_OT_remove_doubles": "Удаление дублей",
    "MESH_OT_flip_normals": "Разворот нормалей",
    "MESH_OT_normals_make_consistent": "Пересчёт нормалей",
    "MESH_OT_separate": "Разделение меша",
    "MESH_OT_fill": "Заливка",
    "MESH_OT_edge_face_add": "Создание грани",
    "MESH_OT_bridge_edge_loops": "Мост между петлями",
    "MESH_OT_mark_seam": "Шов UV",
    "MESH_OT_mark_sharp": "Острое ребро",
    "MESH_OT_select_all": "Выделить всё / снять",
    "MESH_OT_select_linked": "Выделение связанного",
    "MESH_OT_loop_select": "Выделение петли",
    "MESH_OT_select_mode": "Режим выделения",
    "MESH_OT_spin": "Вращение (Spin)",
    "MESH_OT_duplicate_move": "Дублирование",
    "MESH_OT_symmetrize": "Симметризация",
    "VIEW3D_OT_select": "Выделение",
    "VIEW3D_OT_select_box": "Выделение рамкой",
    "VIEW3D_OT_select_circle": "Выделение кругом",
    "VIEW3D_OT_select_lasso": "Выделение лассо",
    "OBJECT_OT_select_all": "Выделить всё / снять",
    "OBJECT_OT_editmode_toggle": "Вход / выход из редактирования",
    "OBJECT_OT_mode_set": "Смена режима",
    "OBJECT_OT_delete": "Удаление объектов",
    "OBJECT_OT_duplicate_move": "Дублирование",
    "OBJECT_OT_duplicate_move_linked": "Связанный дубликат",
    "OBJECT_OT_join": "Объединение объектов",
    "OBJECT_OT_modifier_add": "Добавлен модификатор",
    "OBJECT_OT_modifier_apply": "Применён модификатор",
    "OBJECT_OT_modifier_remove": "Удалён модификатор",
    "OBJECT_OT_shade_smooth": "Сглаживание (Shade Smooth)",
    "OBJECT_OT_shade_flat": "Плоское затенение",
    "OBJECT_OT_shade_auto_smooth": "Auto Smooth",
    "OBJECT_OT_transform_apply": "Применение трансформаций",
    "OBJECT_OT_origin_set": "Смена точки опоры",
    "OBJECT_OT_parent_set": "Назначен родитель",
    "OBJECT_OT_parent_clear": "Снят родитель",
    "OBJECT_OT_material_slot_add": "Добавлен слот материала",
    "OBJECT_OT_convert": "Конвертация",
    "OBJECT_OT_bake": "Запекание",
    "UV_OT_unwrap": "UV-развёртка",
    "UV_OT_smart_project": "Smart UV Project",
    "UV_OT_pack_islands": "Упаковка UV-островов",
    "SCULPT_OT_brush_stroke": "Мазок скульпт-кистью",
    "PAINT_OT_image_paint": "Мазок кистью",
    "PAINT_OT_vertex_paint": "Мазок кистью (вершины)",
    "PAINT_OT_weight_paint": "Мазок кистью (веса)",
    "SCULPT_OT_dynamic_topology_toggle": "Dyntopo",
    "SCULPT_OT_symmetrize": "Симметризация скульпта",
    "ANIM_OT_keyframe_insert": "Ключ анимации",
    "ANIM_OT_keyframe_insert_menu": "Ключ анимации",
    "ANIM_OT_keyframe_delete": "Удалён ключ",
    "OUTLINER_OT_item_activate": "Выбор в аутлайнере",
    "NODE_OT_add_node": "Добавлена нода",
    "NODE_OT_link": "Соединение нод",
    "NODE_OT_delete": "Удаление нод",
    "TK_OT_quad_draw": "Quad Draw",
    "TK_OT_retopo_relax": "Расслабить и прижать",
    "TK_OT_layer_add": "Добавлен слой текстуры",
}

AXES = "XYZ"


def _primitive(idname):
    if idname.startswith("MESH_OT_primitive_"):
        kind = idname[len("MESH_OT_primitive_"):].replace("_add", "").replace("_", " ")
        return f"Добавлен примитив: {kind}"
    return None


def _vec(v):
    return "(" + ", ".join(f"{x:.3g}" for x in v) + ")"


def _details(idname, props):
    def g(name, default=None):
        try:
            return getattr(props, name)
        except Exception:
            return default

    if idname in ("MESH_OT_extrude_region_move", "MESH_OT_duplicate_move",
                  "OBJECT_OT_duplicate_move", "MESH_OT_extrude_context_move"):
        inner = g("TRANSFORM_OT_translate")
        if inner is not None:
            return _details("TRANSFORM_OT_translate", inner)
    if idname == "MESH_OT_loopcut_slide":
        inner = g("MESH_OT_loopcut")
        if inner is not None:
            return f"разрезов: {inner.number_cuts}"
    if idname in ("TRANSFORM_OT_translate",):
        v = g("value")
        ca = g("constraint_axis")
        if v is None:
            return ""
        if ca is not None and sum(bool(x) for x in ca) == 1:
            i = [bool(x) for x in ca].index(True)
            return f"по {AXES[i]} на {C.fmt_len(v[i])}"
        return f"на {_vec(v)}"
    if idname == "TRANSFORM_OT_rotate":
        v = g("value")
        ax = g("orient_axis", "")
        if v is None:
            return ""
        return f"{math.degrees(v):.1f}° вокруг {ax}"
    if idname == "TRANSFORM_OT_resize":
        v = g("value")
        if v is None:
            return ""
        if abs(v[0] - v[1]) < 1e-6 and abs(v[1] - v[2]) < 1e-6:
            return f"×{v[0]:.3g}"
        return f"×{_vec(v)}"
    if idname == "MESH_OT_bevel":
        return f"ширина {C.fmt_len(g('offset', 0))}, сегментов {g('segments', 1)}"
    if idname == "MESH_OT_inset":
        return f"толщина {C.fmt_len(g('thickness', 0))}"
    if idname == "MESH_OT_subdivide":
        return f"разрезов: {g('number_cuts', 1)}"
    if idname in ("OBJECT_OT_modifier_add",):
        return str(g("type", ""))
    if idname in ("OBJECT_OT_modifier_apply", "OBJECT_OT_modifier_remove"):
        return str(g("modifier", ""))
    if idname == "OBJECT_OT_mode_set":
        return str(g("mode", ""))
    if idname == "MESH_OT_delete":
        return {"VERT": "вершины", "EDGE": "рёбра", "FACE": "грани"}.get(g("type", ""), "")
    if idname == "MESH_OT_select_all":
        return {"SELECT": "выделить", "DESELECT": "снять", "INVERT": "инверсия",
                "TOGGLE": "переключить"}.get(g("action", ""), "")
    return ""


def describe(op):
    idname = op.bl_idname
    name = NAMES.get(idname) or _primitive(idname) or op.bl_label
    try:
        det = _details(idname, op.properties)
    except Exception:
        det = ""
    return name, det


def _record(op):
    ctx = bpy.context
    name, det = describe(op)
    obj = getattr(ctx, "active_object", None)
    entry = {
        "name": name,
        "detail": det,
        "obj": obj.name if obj else "",
        "mode": getattr(ctx, "mode", ""),
        "time": time.strftime("%H:%M:%S"),
        "idname": op.bl_idname,
    }
    del _log[_ptr["i"] + 1:]
    _log.append(entry)
    if len(_log) > MAX_LOG:
        del _log[0]
    _ptr["i"] = len(_log) - 1


def _poll_ops():
    wm = bpy.context.window_manager
    if wm is None:
        return 0.5
    ops = wm.operators
    if len(ops):
        op = ops[-1]
        p = op.as_pointer()
        if p != _ptr["last"]:
            _ptr["last"] = p
            if op.bl_idname not in SKIP and "UNDO" in op.bl_options:
                _record(op)
                C.tag_redraw_3d()
                _redraw_ui()
    return 0.25


def _redraw_ui():
    wm = bpy.context.window_manager
    if not wm:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                for r in area.regions:
                    if r.type == "UI":
                        r.tag_redraw()


@bpy.app.handlers.persistent
def _on_undo(*_a):
    if _ptr["i"] >= 0:
        _ptr["i"] -= 1
    _sync_last()
    _redraw_ui()


@bpy.app.handlers.persistent
def _on_redo(*_a):
    if _ptr["i"] < len(_log) - 1:
        _ptr["i"] += 1
    _sync_last()
    _redraw_ui()


def _sync_last():
    wm = bpy.context.window_manager
    if wm and len(wm.operators):
        _ptr["last"] = wm.operators[-1].as_pointer()


@bpy.app.handlers.persistent
def _on_load(*_a):
    _log.clear()
    _ptr["i"] = -1
    _sync_last()
    if not bpy.app.timers.is_registered(_poll_ops):
        bpy.app.timers.register(_poll_ops, first_interval=0.5, persistent=True)


def entry_text(e):
    t = e["name"]
    if e["detail"]:
        t += " · " + e["detail"]
    if e["obj"]:
        t += " · " + e["obj"]
    return t


def _draw_hud():
    try:
        if not bpy.context.scene.tk_history_hud:
            return
    except Exception:
        return
    region = bpy.context.region
    if region is None:
        return
    lines = []
    i = _ptr["i"]
    if 0 <= i < len(_log):
        lines.append(("Ctrl+Z отменит: " + entry_text(_log[i]), (0.95, 0.95, 0.95, 0.9)))
    if i + 1 < len(_log):
        lines.append(("Ctrl+Shift+Z вернёт: " + entry_text(_log[i + 1]), (0.7, 0.85, 1.0, 0.8)))
    if lines:
        C.draw_text_lines(lines, 18, 46, size=12, line_h=17)


# ---------------------------------------------------------------------------
# Operators & UI
# ---------------------------------------------------------------------------

def _run_steps(steps, win_ptr):
    wm = bpy.context.window_manager
    win = next((w for w in wm.windows if w.as_pointer() == win_ptr), None) or wm.windows[0]
    area = next((a for a in win.screen.areas if a.type == "VIEW_3D"), win.screen.areas[0])
    with bpy.context.temp_override(window=win, screen=win.screen, area=area):
        for _ in range(abs(steps)):
            try:
                if steps > 0:
                    bpy.ops.ed.undo()
                else:
                    bpy.ops.ed.redo()
            except RuntimeError:
                break
    return None


class TK_OT_history_goto(bpy.types.Operator):
    bl_idname = "tk.history_goto"
    bl_label = "Вернуться к этому шагу"
    bl_description = "Отменить (или вернуть) действия до выбранного шага"

    index: IntProperty()

    def execute(self, context):
        steps = _ptr["i"] - self.index
        if steps == 0:
            return {"CANCELLED"}
        wp = context.window.as_pointer() if context.window else 0
        bpy.app.timers.register(lambda: _run_steps(steps, wp), first_interval=0.01)
        return {"FINISHED"}


class TK_OT_history_clear(bpy.types.Operator):
    bl_idname = "tk.history_clear"
    bl_label = "Очистить журнал"
    bl_description = "Очистить список (саму историю отмены Blender это не трогает)"

    def execute(self, context):
        _log.clear()
        _ptr["i"] = -1
        return {"FINISHED"}


class TK_PT_history(bpy.types.Panel):
    bl_label = "История действий"
    bl_idname = "TK_PT_history"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "tk_history_hud")
        i = _ptr["i"]
        if 0 <= i < len(_log):
            box = layout.box()
            box.label(text="Ctrl+Z отменит:", icon="LOOP_BACK")
            box.label(text=entry_text(_log[i]))
        if not _log:
            layout.label(text="Пока пусто — действия появятся здесь", icon="INFO")
            return
        col = layout.column(align=True)
        start = len(_log) - 1
        for k in range(start, max(-1, start - 40), -1):
            e = _log[k]
            row = col.row(align=True)
            undone = k > i
            icon = "RIGHTARROW_THIN" if k == i else ("GHOST_DISABLED" if undone else "DOT")
            sub = row.row(align=True)
            sub.active = not undone
            sub.label(text=f"{e['time']}  {entry_text(e)}", icon=icon)
            if k != i:
                op = row.operator("tk.history_goto", text="",
                                  icon="LOOP_FORWARDS" if undone else "LOOP_BACK")
                op.index = k
        layout.operator("tk.history_clear", icon="TRASH")
        layout.label(text="Шаги без оператора (часть мазков, смена кадров) могут не попасть",
                     icon="QUESTION")


classes = (TK_OT_history_goto, TK_OT_history_clear, TK_PT_history)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_history_hud = BoolProperty(
        name="Подсказка «что отменит Ctrl+Z» во вьюпорте", default=True)
    bpy.app.handlers.undo_post.append(_on_undo)
    bpy.app.handlers.redo_post.append(_on_redo)
    bpy.app.handlers.load_post.append(_on_load)
    if not bpy.app.timers.is_registered(_poll_ops):
        bpy.app.timers.register(_poll_ops, first_interval=0.5, persistent=True)
    if C.can_draw():
        _ptr["h2d"] = bpy.types.SpaceView3D.draw_handler_add(_draw_hud, (), "WINDOW", "POST_PIXEL")


def unregister():
    if _ptr["h2d"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_ptr["h2d"], "WINDOW")
        _ptr["h2d"] = None
    if bpy.app.timers.is_registered(_poll_ops):
        bpy.app.timers.unregister(_poll_ops)
    for lst, fn in ((bpy.app.handlers.undo_post, _on_undo),
                    (bpy.app.handlers.redo_post, _on_redo),
                    (bpy.app.handlers.load_post, _on_load)):
        if fn in lst:
            lst.remove(fn)
    del bpy.types.Scene.tk_history_hud
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
