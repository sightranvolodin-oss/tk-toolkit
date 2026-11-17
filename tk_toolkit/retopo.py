"""Retopology helper: one-click setup, a Quad Draw style modal tool, relax + reproject."""

import bpy
import bmesh
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty
from mathutils import Vector
from mathutils.geometry import intersect_line_line_2d

from . import common as C


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _ref_poll(self, obj):
    return obj.type == "MESH"


class TK_RetopoSettings(bpy.types.PropertyGroup):
    reference: PointerProperty(name="Референс", type=bpy.types.Object, poll=_ref_poll,
                               description="High-poly, по которому строится ретопология")
    offset: FloatProperty(name="Отступ над поверхностью", default=0.002, min=0.0, soft_max=0.05,
                          subtype="DISTANCE")
    weld_px: IntProperty(name="Радиус прилипания, px", default=14, min=4, max=60,
                         description="Насколько близко к вершине нужно кликнуть, чтобы использовать её")
    edge_px: IntProperty(name="Радиус поиска ребра, px", default=40, min=8, max=200)


def get_reference(context, obj=None):
    s = context.scene.tk_retopo
    if s.reference:
        return s.reference
    obj = obj or context.active_object
    if obj:
        for m in obj.modifiers:
            if m.type == "SHRINKWRAP" and m.target:
                return m.target
    return None


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

class Projector:
    """Projects points (in the retopo object's local space) onto the reference surface."""

    def __init__(self, ref, retopo, depsgraph):
        self.bvh, self.rm, self.rminv, self.rminv3 = C.object_bvh(ref, depsgraph)
        self.tm = retopo.matrix_world.copy()
        self.tminv = self.tm.inverted_safe()
        self.r_n3 = self.rminv3.transposed()          # ref-local normal -> world
        self.t_n3 = self.tm.to_3x3().transposed()     # world normal -> retopo-local

    def nearest(self, local_co):
        w = self.tm @ local_co
        res = self.bvh.find_nearest(self.rminv @ w)
        if res[0] is None:
            return local_co, None
        hit_w = self.rm @ res[0]
        n_w = (self.r_n3 @ res[1]).normalized()
        return self.tminv @ hit_w, (self.t_n3 @ n_w).normalized()

    def ray(self, origin_w, dir_w):
        r = C.world_ray_bvh(self.bvh, self.rm, self.rminv, self.rminv3, origin_w, dir_w)
        if r is None:
            return None
        hit_w = r[0]
        n_w = (self.r_n3 @ r[1]).normalized()
        return self.tminv @ hit_w, (self.t_n3 @ n_w).normalized(), hit_w


# ---------------------------------------------------------------------------
# Pure bmesh operations (unit-tested headless)
# ---------------------------------------------------------------------------

def _plane_2d(points, normal):
    n = normal.normalized() if normal.length > 0 else Vector((0, 0, 1))
    t = n.orthogonal().normalized()
    b = n.cross(t)
    c = sum(points, Vector()) / len(points)
    return [Vector(((p - c).dot(t), (p - c).dot(b))) for p in points]


def _simple_quad(p2):
    a, b, c, d = p2
    if intersect_line_line_2d(a, b, c, d):
        return False
    if intersect_line_line_2d(b, c, d, a):
        return False
    return True


def order_quad(verts, normal):
    """Reorder 4 verts so they form a non self-intersecting quad."""
    if len(verts) != 4:
        return list(verts)
    a, b, c, d = verts
    for cand in ([a, b, c, d], [a, b, d, c], [a, c, b, d]):
        if _simple_quad(_plane_2d([v.co for v in cand], normal)):
            return cand
    return list(verts)


def make_face(bm, verts, ref_normal=None):
    verts = [v for v in verts if v.is_valid]
    uniq = []
    for v in verts:
        if v not in uniq:
            uniq.append(v)
    if len(uniq) < 3:
        return None
    if len(uniq) == 4:
        n = ref_normal
        if n is None:
            n = (uniq[1].co - uniq[0].co).cross(uniq[2].co - uniq[0].co)
        uniq = order_quad(uniq, n)
    existing = bm.faces.get(uniq)
    if existing:
        return existing
    try:
        f = bm.faces.new(uniq)
    except ValueError:
        return None
    f.normal_update()
    if ref_normal is not None and f.normal.dot(ref_normal) < 0:
        f.normal_flip()
    f.select_set(True)
    return f


def extend_edge(bm, edge, offset, project=None, find_existing=None, ref_normal=None):
    """Build a quad off a boundary edge, translated by *offset* (local space)."""
    v1, v2 = edge.verts
    new = []
    for v in (v1, v2):
        p = v.co + offset
        if project:
            p = project(p)
        nv = find_existing(p, exclude=(v1, v2)) if find_existing else None
        if nv is None:
            nv = bm.verts.new(p)
        new.append(nv)
    n1, n2 = new
    if n1 is n2:
        return make_face(bm, [v1, v2, n1], ref_normal)
    return make_face(bm, [v1, v2, n2, n1], ref_normal)


def relax_verts(bm, verts, iterations=5, strength=0.5, pin_boundary=True, project=None):
    verts = [v for v in verts if v.is_valid]
    for _ in range(max(0, iterations)):
        new_pos = {}
        for v in verts:
            if pin_boundary and (v.is_boundary or v.is_wire):
                continue
            nbr = [e.other_vert(v) for e in v.link_edges]
            if not nbr:
                continue
            avg = sum((n.co for n in nbr), Vector()) / len(nbr)
            new_pos[v] = v.co.lerp(avg, strength)
        for v, p in new_pos.items():
            v.co = p
        if project:
            for v in verts:
                v.co = project(v.co)
    if project and iterations == 0:
        for v in verts:
            v.co = project(v.co)


# ---------------------------------------------------------------------------
# Setup / finish
# ---------------------------------------------------------------------------

def _overlay(enable, offset):
    for sp in C.view3d_spaces():
        ov = sp.overlay
        if hasattr(ov, "show_retopology"):
            ov.show_retopology = enable
            if enable:
                ov.retopology_offset = max(offset, 0.0001)


class TK_OT_retopo_setup(bpy.types.Operator):
    bl_idname = "tk.retopo_setup"
    bl_label = "Начать ретопологию"
    bl_description = ("Создать объект ретопологии поверх активного high-poly: shrinkwrap, "
                      "привязка к поверхности, оверлей ретопологии и режим редактирования")
    bl_options = {"REGISTER", "UNDO"}

    mirror_x: BoolProperty(name="Зеркало по X", default=False)
    lock_reference: BoolProperty(name="Запретить выделение референса", default=True)

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and context.mode == "OBJECT"

    def execute(self, context):
        ref = context.active_object
        s = context.scene.tk_retopo
        s.reference = ref
        me = bpy.data.meshes.new(ref.name + "_retopo")
        obj = bpy.data.objects.new(ref.name + "_retopo", me)
        coll = ref.users_collection[0] if ref.users_collection else context.scene.collection
        coll.objects.link(obj)
        obj.matrix_world = ref.matrix_world.copy()
        if self.mirror_x:
            mm = obj.modifiers.new("Mirror", "MIRROR")
            mm.use_clip = True
            mm.use_mirror_merge = True
        sw = obj.modifiers.new("Retopo Shrinkwrap", "SHRINKWRAP")
        sw.target = ref
        sw.wrap_method = "NEAREST_SURFACEPOINT"
        sw.wrap_mode = "ABOVE_SURFACE"
        sw.offset = s.offset
        sw.show_in_editmode = True
        sw.show_on_cage = True
        obj.show_wire = True
        obj.show_in_front = False
        if self.lock_reference:
            ref.hide_select = True
        for o in context.selected_objects:
            o.select_set(False)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        ts = context.tool_settings
        ts.use_snap = True
        try:
            ts.snap_elements = {"VERTEX", "FACE_PROJECT"}
        except TypeError:
            ts.snap_elements = {"FACE_PROJECT"}
        if hasattr(ts, "use_snap_self"):
            ts.use_snap_self = False
        if hasattr(ts, "use_snap_selectable"):
            ts.use_snap_selectable = False
        _overlay(True, s.offset)
        bpy.ops.object.mode_set(mode="EDIT")
        ts.mesh_select_mode = (True, False, False)
        self.report({"INFO"}, "Ретопология готова: запустите Quad Draw (кнопка в панели)")
        return {"FINISHED"}


class TK_OT_retopo_finish(bpy.types.Operator):
    bl_idname = "tk.retopo_finish"
    bl_label = "Завершить ретопологию"
    bl_description = "Применить shrinkwrap, вернуть выделяемость референса и выключить привязку"
    bl_options = {"REGISTER", "UNDO"}

    apply_shrinkwrap: BoolProperty(name="Применить shrinkwrap", default=True)

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        ref = get_reference(context, obj)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for m in list(obj.modifiers):
            if m.type == "SHRINKWRAP" and m.name.startswith("Retopo"):
                if self.apply_shrinkwrap:
                    with context.temp_override(object=obj, active_object=obj):
                        try:
                            bpy.ops.object.modifier_apply(modifier=m.name)
                        except RuntimeError:
                            obj.modifiers.remove(m)
                else:
                    obj.modifiers.remove(m)
        if ref:
            ref.hide_select = False
        context.tool_settings.use_snap = False
        _overlay(False, 0)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Relax / reproject
# ---------------------------------------------------------------------------

class TK_OT_retopo_relax(bpy.types.Operator):
    bl_idname = "tk.retopo_relax"
    bl_label = "Расслабить и прижать"
    bl_description = ("Выровнять сетку (усреднение по соседям) и снова прижать вершины к "
                      "поверхности референса. Без выделения — вся сетка")
    bl_options = {"REGISTER", "UNDO"}

    iterations: IntProperty(name="Итераций", default=5, min=0, max=100)
    strength: FloatProperty(name="Сила", default=0.5, min=0.0, max=1.0)
    pin_boundary: BoolProperty(name="Не двигать края", default=True)

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and context.mode == "EDIT_MESH"

    def execute(self, context):
        obj = context.active_object
        ref = get_reference(context, obj)
        if ref is None:
            self.report({"WARNING"}, "Не задан референс (high-poly)")
            return {"CANCELLED"}
        proj = Projector(ref, obj, context.evaluated_depsgraph_get())
        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v for v in bm.verts if v.select] or list(bm.verts)
        relax_verts(bm, verts, self.iterations, self.strength, self.pin_boundary,
                    lambda p: proj.nearest(p)[0])
        bmesh.update_edit_mesh(obj.data)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Quad Draw modal
# ---------------------------------------------------------------------------

_qd = {"handle": None, "op": None}


def _qd_draw():
    op = _qd.get("op")
    if op is None or not C.can_draw():
        return
    try:
        op.draw_preview()
    except ReferenceError:
        _qd["op"] = None


HELP = ("Quad Draw · ЛКМ — точка (4 точки = квад) · Shift+ЛКМ — квад от ближайшего края · "
        "Ctrl+ЛКМ — удалить грань · Enter — грань из точек · Backspace — убрать точку · "
        "Esc/ПКМ — выход")


class TK_OT_quad_draw(bpy.types.Operator):
    bl_idname = "tk.quad_draw"
    bl_label = "Quad Draw"
    bl_description = "Рисовать квады прямо по поверхности референса"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return (o is not None and o.type == "MESH" and context.mode == "EDIT_MESH"
                and context.area and context.area.type == "VIEW_3D")

    # --- helpers -----------------------------------------------------------
    def _screen(self, local_co):
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        return location_3d_to_region_2d(self.region, self.rv3d, self.obj.matrix_world @ local_co)

    def _view_dir(self):
        return self.rv3d.view_rotation @ Vector((0, 0, -1))

    def _nearest_vert(self, coord, exclude=()):
        best, bd = None, self.s.weld_px
        vd = self._view_dir()
        n3 = self.obj.matrix_world.to_3x3()
        for v in self.bm.verts:
            if v in exclude or v.hide:
                continue
            if v.link_faces and (n3 @ v.normal).dot(vd) > 0.35:
                continue  # vertex on the far side of the mesh
            p = self._screen(v.co)
            if p is None:
                continue
            d = (p - coord).length
            if d < bd:
                best, bd = v, d
        return best

    def _nearest_boundary_edge(self, coord):
        best, bd = None, self.s.edge_px
        for e in self.bm.edges:
            if e.hide or not (e.is_boundary or e.is_wire):
                continue
            a = self._screen(e.verts[0].co)
            b = self._screen(e.verts[1].co)
            if a is None or b is None:
                continue
            ab = b - a
            t = 0.0 if ab.length_squared == 0 else max(0.0, min(1.0, (coord - a).dot(ab) / ab.length_squared))
            d = (a + ab * t - coord).length
            if d < bd:
                best, bd = e, d
        return best

    def _surface(self, coord):
        from bpy_extras import view3d_utils
        o = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, coord)
        d = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, coord)
        return self.proj.ray(o, d)

    def _find_existing_for_point(self, local_co, exclude=()):
        p = self._screen(local_co)
        if p is None:
            return None
        return self._nearest_vert(p, exclude)

    def _commit(self, msg):
        bmesh.update_edit_mesh(self.obj.data, loop_triangles=True, destructive=True)
        try:
            bpy.ops.ed.undo_push(message=msg)
        except RuntimeError:
            pass

    # --- drawing -------------------------------------------------------------
    def draw_preview(self):
        mw = self.obj.matrix_world
        pts = [mw @ v.co for v in self.pending if v.is_valid]
        with C.GPUState(depth=False):
            if self.hover_vert is not None and self.hover_vert.is_valid:
                C.draw_batch(C.make_batch("uniform", "POINTS", [mw @ self.hover_vert.co]),
                             (1.0, 0.85, 0.1, 1.0), point_size=12)
            if pts:
                C.draw_batch(C.make_batch("uniform", "POINTS", pts), (0.2, 0.9, 1.0, 1.0),
                             point_size=9)
            loop = list(pts)
            if self.hover_pos is not None and not self.shift:
                loop.append(mw @ self.hover_pos)
            if len(loop) >= 2:
                seg = []
                for i in range(len(loop)):
                    if i + 1 < len(loop):
                        seg += [loop[i], loop[i + 1]]
                if len(loop) >= 3:
                    seg += [loop[-1], loop[0]]
                C.draw_batch(C.make_batch("uniform", "LINES", seg), (0.2, 0.9, 1.0, 0.9), 2.0)
            if self.shift and self.hover_edge is not None and self.hover_pos is not None \
                    and self.hover_edge.is_valid:
                v1, v2 = self.hover_edge.verts
                mid = (v1.co + v2.co) / 2
                off = self.hover_pos - mid
                q = [v1.co, v2.co, v2.co + off, v1.co + off]
                q = [mw @ x for x in q]
                C.draw_batch(C.make_batch("uniform", "LINES",
                                          [q[0], q[1], q[1], q[2], q[2], q[3], q[3], q[0]]),
                             (0.4, 1.0, 0.5, 0.95), 2.0)
            if self.ctrl and self.hover_face is not None and self.hover_face.is_valid:
                f = [mw @ v.co for v in self.hover_face.verts]
                seg = []
                for i in range(len(f)):
                    seg += [f[i], f[(i + 1) % len(f)]]
                C.draw_batch(C.make_batch("uniform", "LINES", seg), (1.0, 0.25, 0.2, 1.0), 3.0)

    # --- lifecycle -----------------------------------------------------------
    def invoke(self, context, event):
        self.obj = context.active_object
        ref = get_reference(context, self.obj)
        if ref is None:
            self.report({"WARNING"}, "Сначала «Начать ретопологию» или укажите референс")
            return {"CANCELLED"}
        self.s = context.scene.tk_retopo
        self.region = context.region
        self.rv3d = context.region_data
        self.proj = Projector(ref, self.obj, context.evaluated_depsgraph_get())
        self.bm = bmesh.from_edit_mesh(self.obj.data)
        self.pending = []
        self.hover_vert = self.hover_edge = self.hover_face = None
        self.hover_pos = self.hover_normal = None
        self.shift = self.ctrl = False
        _qd["op"] = self
        if C.can_draw() and _qd["handle"] is None:
            _qd["handle"] = bpy.types.SpaceView3D.draw_handler_add(_qd_draw, (), "WINDOW", "POST_VIEW")
        context.workspace.status_text_set(HELP)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        for v in self.pending:
            if v.is_valid and not v.link_edges:
                self.bm.verts.remove(v)
        self.pending = []
        bmesh.update_edit_mesh(self.obj.data)
        _qd["op"] = None
        if _qd["handle"] is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_qd["handle"], "WINDOW")
            _qd["handle"] = None
        context.workspace.status_text_set(None)
        context.area.tag_redraw()

    def _update_hover(self, event):
        coord = Vector((event.mouse_region_x, event.mouse_region_y))
        self.shift, self.ctrl = event.shift, event.ctrl
        self.hover_vert = self._nearest_vert(coord, exclude=tuple(self.pending))
        self.hover_edge = self._nearest_boundary_edge(coord) if self.shift else None
        self.hover_face = None
        if self.ctrl:
            best, bd = None, 60
            for f in self.bm.faces:
                p = self._screen(f.calc_center_median())
                if p is not None and (p - coord).length < bd:
                    best, bd = f, (p - coord).length
            self.hover_face = best
        hit = self._surface(coord)
        if hit:
            self.hover_pos, self.hover_normal = hit[0], hit[1]
        else:
            self.hover_pos = self.hover_normal = None
        return coord

    def modal(self, context, event):
        if not self.obj or context.mode != "EDIT_MESH":
            self._finish(context)
            return {"CANCELLED"}
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN",
                          "TRACKPADZOOM", "NDOF_MOTION"} or event.type.startswith("NUMPAD"):
            return {"PASS_THROUGH"}
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}

        if event.type in {"MOUSEMOVE", "LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_CTRL", "RIGHT_CTRL"}:
            self.region = context.region
            self.rv3d = context.region_data
            self._update_hover(event)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "BACK_SPACE" and event.value == "PRESS":
            if self.pending:
                v = self.pending.pop()
                if v.is_valid and not v.link_edges:
                    self.bm.verts.remove(v)
                    bmesh.update_edit_mesh(self.obj.data)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type in {"RET", "NUMPAD_ENTER", "SPACE"} and event.value == "PRESS":
            if len(self.pending) >= 3:
                n = self.hover_normal
                make_face(self.bm, self.pending, n)
                self.pending = []
                self._commit("Quad Draw: грань")
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._update_hover(event)
            if event.ctrl:
                if self.hover_face is not None:
                    bmesh.ops.delete(self.bm, geom=[self.hover_face], context="FACES")
                    self.hover_face = None
                    self._commit("Quad Draw: удалить")
                return {"RUNNING_MODAL"}
            if event.shift:
                if self.hover_edge is not None and self.hover_pos is not None:
                    v1, v2 = self.hover_edge.verts
                    off = self.hover_pos - (v1.co + v2.co) / 2
                    extend_edge(self.bm, self.hover_edge, off,
                                project=lambda p: self.proj.nearest(p)[0],
                                find_existing=self._find_existing_for_point,
                                ref_normal=self.hover_normal)
                    self._commit("Quad Draw: продлить")
                    self._update_hover(event)
                return {"RUNNING_MODAL"}
            v = self.hover_vert
            if v is None and self.hover_pos is not None:
                v = self.bm.verts.new(self.hover_pos)
                self.bm.verts.ensure_lookup_table()
            if v is not None:
                if v not in self.pending:
                    self.pending.append(v)
                bmesh.update_edit_mesh(self.obj.data)
                if len(self.pending) == 4:
                    make_face(self.bm, self.pending, self.hover_normal)
                    self.pending = []
                    self._commit("Quad Draw: квад")
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class TK_PT_retopo(bpy.types.Panel):
    bl_label = "Ретопология"
    bl_idname = "TK_PT_retopo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    def draw(self, context):
        s = context.scene.tk_retopo
        layout = self.layout
        layout.prop(s, "reference")
        if context.mode == "OBJECT":
            layout.operator("tk.retopo_setup", icon="MOD_SHRINKWRAP")
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("tk.quad_draw", icon="MESH_GRID")
        layout.operator("tk.retopo_relax", icon="MOD_SMOOTH")
        layout.operator("tk.retopo_finish", icon="CHECKMARK")
        layout.prop(s, "offset")
        layout.prop(s, "weld_px")
        layout.prop(s, "edge_px")


classes = (
    TK_RetopoSettings,
    TK_OT_retopo_setup,
    TK_OT_retopo_finish,
    TK_OT_retopo_relax,
    TK_OT_quad_draw,
    TK_PT_retopo,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_retopo = bpy.props.PointerProperty(type=TK_RetopoSettings)


def unregister():
    if _qd["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_qd["handle"], "WINDOW")
        _qd["handle"] = None
    del bpy.types.Scene.tk_retopo
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
