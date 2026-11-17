"""Shared helpers: mesh arrays via numpy, GPU drawing, small math utils."""

import bpy
import numpy as np
from mathutils import Matrix, Vector

CATEGORY = "Toolkit"

try:
    import gpu
    from gpu_extras.batch import batch_for_shader
except Exception:  # pragma: no cover - gpu is always present in Blender UI
    gpu = None
    batch_for_shader = None


def can_draw():
    return gpu is not None and not bpy.app.background


# ---------------------------------------------------------------------------
# Mesh data
# ---------------------------------------------------------------------------

class MeshArrays:
    """Snapshot of an evaluated mesh as numpy arrays (local space)."""

    __slots__ = (
        "co", "vn", "tris", "tri_poly", "poly_n", "poly_smooth", "poly_sel",
        "loop_start", "loop_total", "loop_vert", "matrix", "name",
    )

    def world_co(self):
        return transform_points(self.matrix, self.co)

    def world_vn(self):
        return transform_normals(self.matrix, self.vn)

    def world_pn(self):
        return transform_normals(self.matrix, self.poly_n)


def _get(coll, attr, n, dtype, width=1):
    arr = np.empty(n * width, dtype=dtype)
    if n:
        coll.foreach_get(attr, arr)
    return arr.reshape(-1, width) if width > 1 else arr


def mesh_arrays(obj, depsgraph):
    """Read the evaluated mesh of *obj* (with modifiers) into numpy arrays."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        out = MeshArrays()
        nv = len(me.vertices)
        out.co = _get(me.vertices, "co", nv, np.float64, 3)
        if hasattr(me, "vertex_normals"):
            out.vn = _get(me.vertex_normals, "vector", nv, np.float64, 3)
        else:
            out.vn = _get(me.vertices, "normal", nv, np.float64, 3)
        nt = len(me.loop_triangles)
        out.tris = _get(me.loop_triangles, "vertices", nt, np.int64, 3)
        out.tri_poly = _get(me.loop_triangles, "polygon_index", nt, np.int64)
        npoly = len(me.polygons)
        if hasattr(me, "polygon_normals"):
            out.poly_n = _get(me.polygon_normals, "vector", npoly, np.float64, 3)
        else:
            out.poly_n = _get(me.polygons, "normal", npoly, np.float64, 3)
        out.poly_smooth = _get(me.polygons, "use_smooth", npoly, bool)
        out.poly_sel = _get(me.polygons, "select", npoly, bool)
        out.loop_start = _get(me.polygons, "loop_start", npoly, np.int64)
        out.loop_total = _get(me.polygons, "loop_total", npoly, np.int64)
        out.loop_vert = _get(me.loops, "vertex_index", len(me.loops), np.int64)
        out.matrix = obj.matrix_world.copy()
        out.name = obj.name
        return out
    finally:
        ev.to_mesh_clear()


def mat_np(m):
    return np.array([list(r) for r in m], dtype=np.float64)


def transform_points(m, pts):
    a = mat_np(m)
    return pts @ a[:3, :3].T + a[:3, 3]


def transform_normals(m, nrm):
    a = mat_np(m)[:3, :3]
    try:
        n3 = np.linalg.inv(a).T
    except np.linalg.LinAlgError:
        n3 = a
    out = nrm @ n3.T
    return normalize(out)


def normalize(v):
    ln = np.linalg.norm(v, axis=-1, keepdims=True)
    ln[ln == 0] = 1.0
    return v / ln


def polygon_edges(arr):
    """Return (loop_index, next_loop_index, poly_index) for every polygon edge."""
    npoly = len(arr.loop_start)
    if not npoly:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    poly_of_loop = np.repeat(np.arange(npoly), arr.loop_total)
    li = np.arange(len(poly_of_loop))
    start = arr.loop_start[poly_of_loop]
    total = arr.loop_total[poly_of_loop]
    nxt = start + (li - start + 1) % total
    return li, nxt, poly_of_loop


def bbox_diag(points):
    if len(points) == 0:
        return 1.0
    d = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return d if d > 0 else 1.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_shader_cache = {}


def shader(name):
    sh = _shader_cache.get(name)
    if sh is None:
        sh = gpu.shader.from_builtin(name)
        _shader_cache[name] = sh
    return sh


def make_batch(kind, prim, pos, col=None):
    """Create a batch. kind: 'uniform' or 'smooth'. Returns (shader_name, batch)."""
    if not can_draw() or pos is None or len(pos) == 0:
        return None
    pos = np.asarray(pos, dtype=np.float32)
    if kind == "smooth":
        name = "POLYLINE_SMOOTH_COLOR" if prim == "LINES" else "SMOOTH_COLOR"
        if prim == "POINTS":
            name = "FLAT_COLOR"
        data = {"pos": pos, "color": np.asarray(col, dtype=np.float32)}
    else:
        name = "POLYLINE_UNIFORM_COLOR" if prim == "LINES" else "UNIFORM_COLOR"
        if prim == "POINTS":
            name = "UNIFORM_COLOR"
        data = {"pos": pos}
    try:
        sh = shader(name)
        return (name, batch_for_shader(sh, prim, data))
    except Exception:
        fallback = "SMOOTH_COLOR" if kind == "smooth" else "UNIFORM_COLOR"
        sh = shader(fallback)
        return (fallback, batch_for_shader(sh, prim, data))


def draw_batch(item, color=None, line_width=1.0, point_size=6.0):
    if not item:
        return
    name, batch = item
    sh = shader(name)
    sh.bind()
    if name.startswith("POLYLINE"):
        vp = gpu.state.viewport_get()
        sh.uniform_float("viewportSize", (vp[2], vp[3]))
        sh.uniform_float("lineWidth", line_width)
    if color is not None and "UNIFORM" in name:
        sh.uniform_float("color", color)
    gpu.state.point_size_set(point_size)
    batch.draw(sh)


class GPUState:
    """Context manager for common overlay GPU state."""

    def __init__(self, depth=True, blend=True, depth_write=False):
        self.depth = depth
        self.blend = blend
        self.depth_write = depth_write

    def __enter__(self):
        gpu.state.blend_set("ALPHA" if self.blend else "NONE")
        gpu.state.depth_test_set("LESS_EQUAL" if self.depth else "NONE")
        gpu.state.depth_mask_set(self.depth_write)
        return self

    def __exit__(self, *a):
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        gpu.state.point_size_set(1.0)


def draw_text_lines(lines, x, y, size=13, line_h=18):
    """lines: list of (text, (r,g,b,a))."""
    import blf
    fid = 0
    try:
        blf.size(fid, size)
    except TypeError:
        blf.size(fid, size, 72)
    blf.enable(fid, blf.SHADOW)
    blf.shadow(fid, 3, 0.0, 0.0, 0.0, 0.8)
    for i, (text, col) in enumerate(lines):
        blf.color(fid, *col)
        blf.position(fid, x, y - i * line_h, 0)
        blf.draw(fid, text)
    blf.disable(fid, blf.SHADOW)


def tag_redraw_3d():
    wm = bpy.context.window_manager
    if not wm:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def fmt_len(v, scene=None):
    """Format a length using the scene unit settings."""
    try:
        scene = scene or bpy.context.scene
        return bpy.utils.units.to_string(
            scene.unit_settings.system, "LENGTH", v * scene.unit_settings.scale_length,
            precision=4, split_unit=False, compatible_unit=False)
    except Exception:
        return f"{v:.4f}"


def view3d_spaces():
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                for sp in area.spaces:
                    if sp.type == "VIEW_3D":
                        yield sp


def enabled_socket(sockets, name):
    for s in sockets:
        if s.name == name and getattr(s, "enabled", True):
            return s
    for s in sockets:
        if s.identifier == name:
            return s
    return sockets[name]


def selected_objects(view_layer=None):
    """Selected objects, read through bpy.data rather than view_layer.objects.

    Right after bpy.data.objects.remove() inside an operator, Blender 4.x may still keep
    the removed object's base in view_layer.objects until the layer resyncs; iterating that
    list then touches freed memory (crash). Object.select_get() looks the base up by the
    live object instead, which is safe.
    """
    vl = view_layer or bpy.context.view_layer
    out = []
    for o in bpy.data.objects:
        try:
            if o.select_get(view_layer=vl):
                out.append(o)
        except (RuntimeError, ReferenceError):
            pass
    return out


def deselect_all(view_layer=None):
    vl = view_layer or bpy.context.view_layer
    for o in selected_objects(vl):
        try:
            o.select_set(False, view_layer=vl)
        except RuntimeError:
            pass


def world_ray_bvh(bvh, m, minv, minv3, origin, direction, max_dist=None):
    """Cast a world-space ray into a BVH built in object-local space.

    Returns (world_hit, local_normal, world_distance, local_dir) or None.
    The local normal / local dir pair can be used for a back-face test.
    """
    lo = minv @ origin
    ld = minv3 @ direction
    s = ld.length
    if s == 0:
        return None
    ld = ld / s
    if max_dist is None:
        res = bvh.ray_cast(lo, ld)
    else:
        res = bvh.ray_cast(lo, ld, max_dist * s)
    if res[0] is None:
        return None
    return m @ res[0], res[1], res[3] / s, ld


def object_bvh(obj, depsgraph):
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(obj, depsgraph)
    m = obj.matrix_world.copy()
    minv = m.inverted_safe()
    return bvh, m, minv, minv.to_3x3()
