"""Precision tools: live dimensions in the viewport and exact length / distance / angle edits."""

import math

import bpy
import bmesh
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from mathutils import Matrix, Vector

from . import common as C

_h = {"view": None, "pix": None, "cache": None}


# ---------------------------------------------------------------------------
# Measurement data
# ---------------------------------------------------------------------------

def _aabb_world(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def aabb_gap(a, b):
    """Shortest distance between two axis-aligned boxes (0 when they overlap)."""
    (amn, amx), (bmn, bmx) = a, b
    d = Vector((max(0.0, bmn[i] - amx[i], amn[i] - bmx[i]) for i in range(3)))
    return d.length


def angle_at(a, b, c):
    u, v = a - b, c - b
    if u.length == 0 or v.length == 0:
        return 0.0
    return u.angle(v)


def measurements(context):
    """Return (segments_world, labels[(world_pos, text)])."""
    segs, labels = [], []
    obj = context.active_object
    if context.mode == "EDIT_MESH" and obj and obj.type == "MESH":
        me = obj.data
        nsel = me.total_vert_sel
        if nsel == 0 or nsel > 400:
            return segs, labels
        bm = bmesh.from_edit_mesh(me)
        mw = obj.matrix_world
        sel = [v for v in bm.verts if v.select]
        hist = [e for e in bm.select_history if isinstance(e, bmesh.types.BMVert)]
        if len(sel) == 2:
            a, b = (mw @ sel[0].co), (mw @ sel[1].co)
            d = b - a
            segs += [a, b]
            labels.append(((a + b) / 2,
                           f"{C.fmt_len(d.length)}   ΔX {C.fmt_len(abs(d.x))}  "
                           f"ΔY {C.fmt_len(abs(d.y))}  ΔZ {C.fmt_len(abs(d.z))}"))
        elif len(sel) == 3:
            if len(hist) == 3:
                A, B, Cc = (mw @ v.co for v in hist)
            else:
                A, B, Cc = (mw @ v.co for v in sel)
            segs += [A, B, B, Cc]
            labels.append((B, f"∠ {math.degrees(angle_at(A, B, Cc)):.2f}°"))
            labels.append(((A + B) / 2, C.fmt_len((A - B).length)))
            labels.append(((B + Cc) / 2, C.fmt_len((Cc - B).length)))
        else:
            edges = [e for e in bm.edges if e.select]
            if 0 < len(edges) <= 150:
                for e in edges:
                    a, b = (mw @ e.verts[0].co), (mw @ e.verts[1].co)
                    labels.append(((a + b) / 2, C.fmt_len((b - a).length)))
    elif context.mode == "OBJECT":
        sel = context.selected_objects
        if len(sel) == 2:
            a, b = sel[0], sel[1]
            pa, pb = a.matrix_world.translation, b.matrix_world.translation
            segs += [pa.copy(), pb.copy()]
            gap = aabb_gap(_aabb_world(a), _aabb_world(b))
            labels.append(((pa + pb) / 2,
                           f"центры {C.fmt_len((pb - pa).length)} · зазор {C.fmt_len(gap)}"))
    return segs, labels


def _draw_view():
    try:
        if not bpy.context.scene.tk_measure:
            return
        segs, labels = measurements(bpy.context)
    except Exception:
        return
    _h["cache"] = labels
    if segs:
        with C.GPUState(depth=False):
            C.draw_batch(C.make_batch("uniform", "LINES", segs), (1.0, 0.85, 0.2, 0.95), 2.0)


def _draw_pix():
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    try:
        if not bpy.context.scene.tk_measure:
            return
    except Exception:
        return
    labels = _h.get("cache") or []
    region, rv3d = bpy.context.region, bpy.context.region_data
    if not region or not rv3d:
        return
    for pos, text in labels:
        p = location_3d_to_region_2d(region, rv3d, pos)
        if p is not None:
            C.draw_text_lines([(text, (1.0, 0.92, 0.45, 1.0))], p.x + 8, p.y + 8, size=12)


# ---------------------------------------------------------------------------
# Edit operators
# ---------------------------------------------------------------------------

def _edit_poll(context):
    o = context.active_object
    return o is not None and o.type == "MESH" and context.mode == "EDIT_MESH"


def _active_vert(bm):
    a = bm.select_history.active
    return a if isinstance(a, bmesh.types.BMVert) else None


def set_edge_length(bm, mw, edges, length, anchor="CENTER", active=None):
    mwi = mw.inverted_safe()
    for e in edges:
        v1, v2 = e.verts
        a, b = mw @ v1.co, mw @ v2.co
        d = b - a
        if d.length == 0:
            continue
        d.normalize()
        if anchor == "ACTIVE" and active in (v1, v2):
            if active is v1:
                v2.co = mwi @ (a + d * length)
            else:
                v1.co = mwi @ (b - d * length)
        else:
            mid = (a + b) / 2
            v1.co = mwi @ (mid - d * length / 2)
            v2.co = mwi @ (mid + d * length / 2)


def set_angle(mw, A, B, Cv, angle):
    """Rotate vertex C around B (in plane ABC) so that angle ABC == angle."""
    mwi = mw.inverted_safe()
    a, b, c = mw @ A.co, mw @ B.co, mw @ Cv.co
    u, v = a - b, c - b
    axis = u.cross(v)
    if axis.length < 1e-12:
        axis = u.orthogonal()
    axis.normalize()
    cur = u.angle(v) if u.length and v.length else 0.0
    rot = Matrix.Rotation(angle - cur, 3, axis)
    Cv.co = mwi @ (b + rot @ v)


class TK_OT_set_edge_length(bpy.types.Operator):
    bl_idname = "tk.set_edge_length"
    bl_label = "Точная длина ребра"
    bl_description = "Задать выделенным рёбрам точную длину (в мировых единицах)"
    bl_options = {"REGISTER", "UNDO"}

    length: FloatProperty(name="Длина", default=1.0, min=0.0, subtype="DISTANCE", precision=4)
    anchor: EnumProperty(name="Неподвижно", items=[
        ("CENTER", "Центр ребра", ""),
        ("ACTIVE", "Активная вершина", ""),
    ], default="CENTER")

    @classmethod
    def poll(cls, context):
        return _edit_poll(context)

    def invoke(self, context, event):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        edges = [e for e in bm.edges if e.select]
        if not edges:
            self.report({"WARNING"}, "Выделите ребро")
            return {"CANCELLED"}
        mw = obj.matrix_world
        self.length = (mw @ edges[0].verts[1].co - mw @ edges[0].verts[0].co).length
        if _active_vert(bm):
            self.anchor = "ACTIVE"
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        edges = [e for e in bm.edges if e.select]
        set_edge_length(bm, obj.matrix_world, edges, self.length, self.anchor, _active_vert(bm))
        bmesh.update_edit_mesh(obj.data)
        return {"FINISHED"}


class TK_OT_set_distance(bpy.types.Operator):
    bl_idname = "tk.set_distance"
    bl_label = "Точное расстояние"
    bl_description = ("Две выделенные вершины: сдвинуть неактивную вдоль линии между ними, "
                      "чтобы расстояние стало точным")
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="Расстояние", default=1.0, min=0.0, subtype="DISTANCE", precision=4)
    symmetric: BoolProperty(name="Двигать обе симметрично", default=False)

    @classmethod
    def poll(cls, context):
        return _edit_poll(context) and context.active_object.data.total_vert_sel == 2

    def invoke(self, context, event):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        a, b = [v for v in bm.verts if v.select]
        mw = obj.matrix_world
        self.distance = (mw @ a.co - mw @ b.co).length
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        vs = [v for v in bm.verts if v.select]
        if len(vs) != 2:
            return {"CANCELLED"}
        act = _active_vert(bm)
        fixed, moving = (act, vs[0] if vs[1] is act else vs[1]) if act in vs else (vs[0], vs[1])
        mw = obj.matrix_world
        mwi = mw.inverted_safe()
        a, b = mw @ fixed.co, mw @ moving.co
        d = b - a
        if d.length == 0:
            return {"CANCELLED"}
        d.normalize()
        if self.symmetric:
            mid = (a + b) / 2
            fixed.co = mwi @ (mid - d * self.distance / 2)
            moving.co = mwi @ (mid + d * self.distance / 2)
        else:
            moving.co = mwi @ (a + d * self.distance)
        bmesh.update_edit_mesh(obj.data)
        return {"FINISHED"}


class TK_OT_set_angle(bpy.types.Operator):
    bl_idname = "tk.set_angle"
    bl_label = "Точный угол"
    bl_description = ("Выделите по очереди 3 вершины: A, вершину угла B, затем C. "
                      "C повернётся вокруг B до заданного угла")
    bl_options = {"REGISTER", "UNDO"}

    angle: FloatProperty(name="Угол", default=math.radians(90), min=0.0, max=math.pi,
                         subtype="ANGLE")

    @classmethod
    def poll(cls, context):
        return _edit_poll(context) and context.active_object.data.total_vert_sel == 3

    def _verts(self, bm):
        hist = [e for e in bm.select_history if isinstance(e, bmesh.types.BMVert)]
        return hist if len(hist) == 3 else None

    def invoke(self, context, event):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        vs = self._verts(bm)
        if vs is None:
            self.report({"WARNING"}, "Выделяйте 3 вершины по одной (Shift+клик), порядок важен")
            return {"CANCELLED"}
        mw = obj.matrix_world
        self.angle = angle_at(*(mw @ v.co for v in vs))
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        vs = self._verts(bm)
        if vs is None:
            return {"CANCELLED"}
        set_angle(obj.matrix_world, vs[0], vs[1], vs[2], self.angle)
        bmesh.update_edit_mesh(obj.data)
        return {"FINISHED"}


class TK_OT_point_on_edge(bpy.types.Operator):
    bl_idname = "tk.point_on_edge"
    bl_label = "Точка на ребре"
    bl_description = "Добавить вершину на выделенном ребре на точном расстоянии от активной вершины"
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="От активной вершины", default=0.1, min=0.0,
                            subtype="DISTANCE", precision=4)

    @classmethod
    def poll(cls, context):
        return _edit_poll(context)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        edges = [e for e in bm.edges if e.select]
        act = _active_vert(bm)
        if len(edges) != 1:
            self.report({"WARNING"}, "Выделите одно ребро (кликом по вершине задайте начало)")
            return {"CANCELLED"}
        e = edges[0]
        start = act if act in e.verts else e.verts[0]
        mw = obj.matrix_world
        L = (mw @ e.verts[1].co - mw @ e.verts[0].co).length
        if L == 0 or self.distance >= L:
            self.report({"WARNING"}, f"Расстояние должно быть меньше длины ребра ({C.fmt_len(L)})")
            return {"CANCELLED"}
        fac = self.distance / L
        _ne, nv = bmesh.utils.edge_split(e, start, fac)
        for v in bm.verts:
            v.select = False
        for ed in bm.edges:
            ed.select = False
        nv.select = True
        bm.select_history.clear()
        bm.select_history.add(nv)
        bmesh.update_edit_mesh(obj.data)
        return {"FINISHED"}


def quantize(v, step):
    return Vector(round(x / step) * step for x in v)


class TK_OT_quantize(bpy.types.Operator):
    bl_idname = "tk.quantize"
    bl_label = "Округлить к шагу"
    bl_description = ("Привязать выделенные вершины (или объекты) к сетке с заданным шагом "
                      "в мировых координатах")
    bl_options = {"REGISTER", "UNDO"}

    step: FloatProperty(name="Шаг", default=0.01, min=1e-6, subtype="DISTANCE", precision=4)

    def invoke(self, context, event):
        self.step = context.scene.tk_snap_step
        return self.execute(context)

    def execute(self, context):
        step = self.step
        if context.mode == "EDIT_MESH":
            obj = context.active_object
            bm = bmesh.from_edit_mesh(obj.data)
            mw = obj.matrix_world
            mwi = mw.inverted_safe()
            n = 0
            for v in bm.verts:
                if v.select:
                    v.co = mwi @ quantize(mw @ v.co, step)
                    n += 1
            bmesh.update_edit_mesh(obj.data)
        else:
            n = 0
            for o in context.selected_objects:
                o.matrix_world.translation = quantize(o.matrix_world.translation, step)
                n += 1
        self.report({"INFO"}, f"Привязано к шагу {C.fmt_len(step)}: {n}")
        return {"FINISHED"}


class TK_OT_apply_snap_step(bpy.types.Operator):
    bl_idname = "tk.apply_snap_step"
    bl_label = "Сетка с этим шагом"
    bl_description = "Показать сетку с заданным шагом и включить привязку к ней"

    def execute(self, context):
        step = context.scene.tk_snap_step
        ts = context.tool_settings
        ts.use_snap = True
        try:
            ts.snap_elements = {"INCREMENT"}
        except TypeError:
            pass
        if hasattr(ts, "use_snap_grid_absolute"):
            ts.use_snap_grid_absolute = True
        for sp in C.view3d_spaces():
            sp.overlay.grid_scale = step
            sp.overlay.grid_subdivisions = 10
        self.report({"INFO"}, f"Сетка {C.fmt_len(step)}; для гарантии используйте «Округлить к шагу»")
        return {"FINISHED"}


class TK_PT_precision(bpy.types.Panel):
    bl_label = "Точность"
    bl_idname = "TK_PT_precision"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "tk_measure", toggle=True, icon="DRIVER_DISTANCE")
        col = layout.column(align=True)
        col.operator("tk.set_edge_length", icon="ARROW_LEFTRIGHT")
        col.operator("tk.set_distance", icon="CON_DISTLIMIT")
        col.operator("tk.set_angle", icon="DRIVER_ROTATIONAL_DIFFERENCE")
        col.operator("tk.point_on_edge", icon="VERTEXSEL")
        box = layout.box()
        box.prop(context.scene, "tk_snap_step")
        r = box.row(align=True)
        r.operator("tk.quantize", icon="SNAP_INCREMENT")
        r.operator("tk.apply_snap_step", text="", icon="GRID")
        layout.label(text="Размеры: 2 вершины, 3 вершины (угол), рёбра,", icon="INFO")
        layout.label(text="или 2 объекта в Object Mode")


classes = (TK_OT_set_edge_length, TK_OT_set_distance, TK_OT_set_angle, TK_OT_point_on_edge,
           TK_OT_quantize, TK_OT_apply_snap_step, TK_PT_precision)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_measure = BoolProperty(name="Размеры во вьюпорте", default=False)
    bpy.types.Scene.tk_snap_step = FloatProperty(name="Шаг", default=0.01, min=1e-6,
                                                 subtype="DISTANCE", precision=4)
    if C.can_draw():
        _h["view"] = bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), "WINDOW", "POST_VIEW")
        _h["pix"] = bpy.types.SpaceView3D.draw_handler_add(_draw_pix, (), "WINDOW", "POST_PIXEL")


def unregister():
    for k in ("view", "pix"):
        if _h[k] is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_h[k], "WINDOW")
            _h[k] = None
    del bpy.types.Scene.tk_snap_step
    del bpy.types.Scene.tk_measure
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
