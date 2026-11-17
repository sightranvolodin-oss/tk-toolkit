"""Bake preview: see what a Selected-to-Active bake will capture before baking.

Mirrors the Cycles ray setup:
  * without a cage object the ray starts at  P + N * cage_extrusion  and goes along -N
    (N = interpolated vertex normal on smooth faces, face normal on flat faces,
    which is why the "shell" tears apart on hard edges);
  * with a cage object the ray starts on the cage and points to the low-poly surface;
  * max_ray_distance limits the ray length measured from its start (0 = unlimited);
  * the nearest hit over all selected high-poly objects wins.
"""

import time
import math

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, IntProperty, EnumProperty
from mathutils import Vector
from mathutils.interpolate import poly_3d_calc

from . import common as C

STATUS_OK, STATUS_MISS, STATUS_BACK, STATUS_WARN = 0, 1, 2, 3
STATUS_COLORS = {
    STATUS_OK: (0.25, 0.95, 0.35, 0.9),
    STATUS_MISS: (1.0, 0.18, 0.15, 0.95),
    STATUS_BACK: (1.0, 0.55, 0.05, 0.95),
    STATUS_WARN: (1.0, 0.9, 0.15, 0.95),
}

_state = {
    "h3d": None,
    "h2d": None,
    "batches": {},
    "result": None,
    "error": "",
    "dirty": True,
    "sig": None,
    "timer": False,
    "bvh": {},        # name -> (geo_version, (bvh, m, minv, minv3))
    "geo_ver": {},    # name -> int
    "probe": None,
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _mark_dirty(self, context):
    _state["dirty"] = True
    C.tag_redraw_3d()


def _toggle(self, context):
    if self.enabled:
        enable_overlay()
    else:
        disable_overlay()


class TK_BakePreviewSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Просмотр запекания", default=False, update=_toggle)
    samples: IntProperty(
        name="Лучей", default=3000, min=50, max=200000, update=_mark_dirty,
        description="Сколько точек на low-poly проверять. Больше — точнее, но медленнее")
    show_shell: BoolProperty(name="Оболочка (Extrusion / Cage)", default=True, update=_mark_dirty)
    show_limit: BoolProperty(name="Предел Max Ray Distance", default=True, update=_mark_dirty)
    show_rays: BoolProperty(name="Лучи", default=True, update=_mark_dirty)
    show_hits: BoolProperty(name="Точки попадания на high-poly", default=True, update=_mark_dirty)
    show_heat: BoolProperty(name="Карта покрытия на low-poly", default=True, update=_mark_dirty)
    show_hud: BoolProperty(name="Сводка во вьюпорте", default=True, update=_mark_dirty)
    color_by_object: BoolProperty(
        name="Цвет попаданий по объекту", default=False, update=_mark_dirty,
        description="Красить точки попадания цветом high-poly объекта, в который попал луч")
    only_problems: BoolProperty(
        name="Только проблемные лучи", default=False, update=_mark_dirty,
        description="Рисовать только промахи и подозрительные лучи")
    only_selected: BoolProperty(
        name="Только выделенные грани", default=False, update=_mark_dirty,
        description="В режиме редактирования low-poly проверять только выделенную зону")
    shell_alpha: FloatProperty(name="Прозрачность оболочки", default=0.12, min=0.0, max=1.0,
                               update=_mark_dirty)
    auto_update: BoolProperty(name="Обновлять автоматически", default=True)


# ---------------------------------------------------------------------------
# Inputs / BVH cache
# ---------------------------------------------------------------------------

HIGH_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}


def gather(scene, view_layer):
    low = view_layer.objects.active
    if low is None or low.type != "MESH":
        return None, None, "Сделайте активным low-poly меш"
    highs = [o for o in view_layer.objects
             if o is not low and o.type in HIGH_TYPES and o.select_get() and o.visible_get()]
    bake = scene.render.bake
    cage = bake.cage_object if bake.use_cage else None
    if cage is not None:
        highs = [o for o in highs if o is not cage]
    if not highs:
        return low, [], "Выделите high-poly, затем low-poly (активным)"
    return low, highs, ""


def _bvh_for(obj, depsgraph):
    ver = _state["geo_ver"].get(obj.name, 0)
    cached = _state["bvh"].get(obj.name)
    m = obj.matrix_world
    if cached and cached[0] == ver:
        bvh, cm, minv, minv3 = cached[1]
        if cm == m:
            return cached[1]
        mm = m.copy()
        mi = mm.inverted_safe()
        data = (bvh, mm, mi, mi.to_3x3())
        _state["bvh"][obj.name] = (ver, data)
        return data
    data = C.object_bvh(obj, depsgraph)
    _state["bvh"][obj.name] = (ver, data)
    return data


# ---------------------------------------------------------------------------
# Core computation (pure; no GPU)
# ---------------------------------------------------------------------------

def _sample_tris(L, co_w, budget, face_mask, rng):
    tris = L.tris
    if len(tris) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3))
    cand = np.arange(len(tris))
    if face_mask is not None:
        cand = cand[face_mask[L.tri_poly]]
    if len(cand) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3))
    a, b, c = co_w[tris[cand, 0]], co_w[tris[cand, 1]], co_w[tris[cand, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        area = np.ones_like(area)
    ids = []
    bary = []
    if len(cand) <= budget:
        # one sample in the middle of every triangle, the rest by area
        ids.append(cand)
        bary.append(np.full((len(cand), 3), 1.0 / 3.0))
        extra = budget - len(cand)
    else:
        extra = budget
    if extra > 0:
        pick = rng.choice(len(cand), size=extra, p=area / area.sum())
        r1 = rng.random(extra)
        r2 = rng.random(extra)
        flip = r1 + r2 > 1.0
        r1[flip] = 1.0 - r1[flip]
        r2[flip] = 1.0 - r2[flip]
        ids.append(cand[pick])
        bary.append(np.stack([1.0 - r1 - r2, r1, r2], axis=1))
    return np.concatenate(ids), np.concatenate(bary)


def _interp(values, tris, ids, bary):
    t = tris[ids]
    return (values[t[:, 0]] * bary[:, 0:1] + values[t[:, 1]] * bary[:, 1:2]
            + values[t[:, 2]] * bary[:, 2:3])


def ray_setup(L, co_w, vn_w, pn_w, ids, bary, extrusion, cage_w=None):
    """Return P (surface), N (bake normal), O (ray origin), D (ray dir)."""
    P = _interp(co_w, L.tris, ids, bary)
    smooth = L.poly_smooth[L.tri_poly[ids]]
    Ns = C.normalize(_interp(vn_w, L.tris, ids, bary))
    Nf = pn_w[L.tri_poly[ids]]
    N = np.where(smooth[:, None], Ns, Nf)
    if cage_w is not None:
        O = _interp(cage_w, L.tris, ids, bary)
        D = P - O
        ln = np.linalg.norm(D, axis=1)
        bad = ln < 1e-9
        D = C.normalize(D)
        D[bad] = -N[bad]
    else:
        O = P + N * extrusion
        D = -N
    return P, N, O, D


def cast_rays(O, D, highs_bvh, limit):
    n = len(O)
    hit = np.zeros(n, bool)
    hp = np.zeros((n, 3))
    dist = np.full(n, np.inf)
    back = np.zeros(n, bool)
    hobj = np.full(n, -1, np.int64)
    for i in range(n):
        o = Vector(O[i])
        d = Vector(D[i])
        best = None
        bk = -1
        for k, (bvh, m, minv, minv3) in enumerate(highs_bvh):
            r = C.world_ray_bvh(bvh, m, minv, minv3, o, d, limit)
            if r is not None and (best is None or r[2] < best[2]):
                best = r
                bk = k
        if best is not None:
            hit[i] = True
            hp[i] = best[0]
            dist[i] = best[2]
            back[i] = best[1].dot(best[3]) > 0.0
            hobj[i] = bk
    return hit, hp, dist, back, hobj


def compute(scene, view_layer, depsgraph, settings):
    low, highs, err = gather(scene, view_layer)
    if err:
        return None, err
    bake = scene.render.bake
    e = float(bake.cage_extrusion)
    d = float(bake.max_ray_distance)
    limit = d if d > 0 else None
    cage_obj = bake.cage_object if (bake.use_cage and bake.cage_object) else None

    t0 = time.perf_counter()
    L = C.mesh_arrays(low, depsgraph)
    co_w, vn_w, pn_w = L.world_co(), L.world_vn(), L.world_pn()
    diag = C.bbox_diag(co_w)

    cage_w = None
    CA = None
    if cage_obj is not None:
        CA = C.mesh_arrays(cage_obj, depsgraph)
        if len(CA.co) != len(L.co):
            return None, (f"Cage «{cage_obj.name}» должен иметь столько же вершин, "
                          f"сколько low-poly ({len(CA.co)} ≠ {len(L.co)})")
        cage_w = CA.world_co()

    face_mask = None
    if settings.only_selected and low.mode == "EDIT":
        face_mask = L.poly_sel
        if not face_mask.any():
            return None, "Нет выделенных граней в low-poly"

    rng = np.random.default_rng(7)
    ids, bary = _sample_tris(L, co_w, int(settings.samples), face_mask, rng)
    P, N, O, D = ray_setup(L, co_w, vn_w, pn_w, ids, bary, e, cage_w)

    highs_bvh = [_bvh_for(h, depsgraph) for h in highs]
    hit, hp, dist, back, hobj = cast_rays(O, D, highs_bvh, limit)

    # classify
    depth = np.einsum("ij,ij->i", hp - P, N)  # >0: high is outside low
    status = np.full(len(P), STATUS_OK, np.int64)
    status[~hit] = STATUS_MISS
    warn = np.zeros(len(P), bool)
    if limit is not None:
        warn |= hit & (dist > 0.9 * limit)
    else:
        thresh = max(3.0 * e, 0.03 * diag)
        warn |= hit & (-depth > thresh)
    status[warn] = STATUS_WARN
    status[hit & back] = STATUS_BACK

    # coverage per triangle
    ntri = len(L.tris)
    cnt = np.bincount(ids, minlength=ntri)
    okc = np.bincount(ids, weights=(status == STATUS_OK).astype(float), minlength=ntri)
    bkc = np.bincount(ids, weights=(status == STATUS_BACK).astype(float), minlength=ntri)
    wrc = np.bincount(ids, weights=(status == STATUS_WARN).astype(float), minlength=ntri)

    flat_faces = int((~L.poly_smooth).sum())
    stats = {
        "n": int(len(P)),
        "ok": int((status == STATUS_OK).sum()),
        "miss": int((status == STATUS_MISS).sum()),
        "back": int((status == STATUS_BACK).sum()),
        "warn": int((status == STATUS_WARN).sum()),
        "per_obj": {h.name: int((hobj == k).sum()) for k, h in enumerate(highs)},
        "flat_faces": flat_faces,
        "extrusion": e,
        "max_dist": d,
        "cage": cage_obj.name if cage_obj else "",
        "low": low.name,
        "highs": [h.name for h in highs],
        "time": 0.0,
        "selected_to_active": bool(bake.use_selected_to_active),
    }
    hints = []
    if not bake.use_selected_to_active:
        hints.append("В настройках Bake выключено Selected to Active")
    if cage_obj is None and e > 0 and flat_faces:
        hints.append(f"{flat_faces} плоских граней: оболочка рвётся на жёстких рёбрах "
                     "(Shade Smooth / Auto Smooth или отдельный Cage)")
    if limit is not None and cage_obj is None and d < e:
        hints.append("Max Ray Distance меньше Extrusion — лучи не доходят до low-poly")
    if stats["back"]:
        hints.append("Есть попадания в изнанку high-poly — проверьте нормали")
    if stats["miss"] and cage_obj is None and stats["miss"] > 0.02 * max(1, stats["n"]):
        hints.append("Много промахов: увеличьте Extrusion или Max Ray Distance (кнопка «Подобрать»)")
    stats["hints"] = hints

    res = {
        "low": L, "co_w": co_w, "vn_w": vn_w, "pn_w": pn_w, "diag": diag,
        "cage": CA, "cage_w": cage_w,
        "ids": ids, "P": P, "N": N, "O": O, "D": D,
        "hit": hit, "hp": hp, "dist": dist, "status": status, "hobj": hobj,
        "tri_cnt": cnt, "tri_ok": okc, "tri_back": bkc, "tri_warn": wrc,
        "extrusion": e, "limit": limit, "stats": stats,
        "highs_bvh": highs_bvh, "high_names": [h.name for h in highs],
    }
    stats["time"] = time.perf_counter() - t0
    return res, ""


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def _corner_normals(L, vn_w, pn_w, verts, polys):
    sm = L.poly_smooth[polys]
    return np.where(sm[:, None], vn_w[verts], pn_w[polys])


def _object_color(k):
    h = (k * 0.61803398875) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return (r, g, b, 0.95)


def build_batches(res, settings):
    B = {}
    if res is None or not C.can_draw():
        return B
    L = res["low"]
    co_w, vn_w, pn_w = res["co_w"], res["vn_w"], res["pn_w"]
    diag = res["diag"]
    e = res["extrusion"]
    eps = diag * 0.0006

    tris = L.tris
    tri_poly = L.tri_poly
    corner_v = tris.reshape(-1)
    corner_p = np.repeat(tri_poly, 3)
    cn = _corner_normals(L, vn_w, pn_w, corner_v, corner_p)

    # coverage heat map on the low-poly
    if settings.show_heat and len(tris):
        cnt = res["tri_cnt"]
        sampled = cnt > 0
        frac = np.zeros(len(tris))
        frac[sampled] = res["tri_ok"][sampled] / cnt[sampled]
        col = np.zeros((len(tris), 4))
        col[:, 0] = 1.0 - frac
        col[:, 1] = 0.25 + 0.7 * frac
        col[:, 2] = 0.15
        col[:, 3] = 0.45
        backheavy = sampled & (res["tri_back"] > 0.5 * cnt)
        col[backheavy] = (1.0, 0.55, 0.05, 0.5)
        col[~sampled] = (0.5, 0.5, 0.5, 0.0)
        keep = np.repeat(sampled, 3)
        pos = co_w[corner_v] + cn * eps
        cols = np.repeat(col, 3, axis=0)
        B["heat"] = C.make_batch("smooth", "TRIS", pos[keep], cols[keep])

    li, nxt, pol = C.polygon_edges(L)
    ev0 = L.loop_vert[li]
    ev1 = L.loop_vert[nxt]

    if settings.show_shell:
        if res["cage_w"] is not None:
            CA = res["cage"]
            cw = res["cage_w"]
            B["shell_tris"] = C.make_batch("uniform", "TRIS", cw[CA.tris.reshape(-1)])
            cli, cnx, _ = C.polygon_edges(CA)
            B["shell_edges"] = C.make_batch(
                "uniform", "LINES",
                np.stack([cw[CA.loop_vert[cli]], cw[CA.loop_vert[cnx]]], axis=1).reshape(-1, 3))
        elif e > 0:
            B["shell_tris"] = C.make_batch("uniform", "TRIS", co_w[corner_v] + cn * e)
            n0 = _corner_normals(L, vn_w, pn_w, ev0, pol)
            n1 = _corner_normals(L, vn_w, pn_w, ev1, pol)
            seg = np.stack([co_w[ev0] + n0 * e, co_w[ev1] + n1 * e], axis=1).reshape(-1, 3)
            B["shell_edges"] = C.make_batch("uniform", "LINES", seg)

    if settings.show_limit and res["limit"] is not None and res["cage_w"] is None:
        off = e - res["limit"]
        n0 = _corner_normals(L, vn_w, pn_w, ev0, pol)
        n1 = _corner_normals(L, vn_w, pn_w, ev1, pol)
        seg = np.stack([co_w[ev0] + n0 * off, co_w[ev1] + n1 * off], axis=1).reshape(-1, 3)
        B["limit_edges"] = C.make_batch("uniform", "LINES", seg)

    status = res["status"]
    mask = np.ones(len(status), bool)
    if settings.only_problems:
        mask = status != STATUS_OK

    if settings.show_rays and mask.any():
        O, D, hit, hp = res["O"], res["D"], res["hit"], res["hp"]
        vis = res["limit"] if res["limit"] is not None else max(2.0 * e, 0.03 * diag)
        end = np.where(hit[:, None], hp, O + D * vis)
        seg = np.stack([O, end], axis=1)[mask].reshape(-1, 3)
        cols = np.array([STATUS_COLORS[s] for s in status[mask]])
        cols[:, 3] *= 0.75
        B["rays"] = C.make_batch("smooth", "LINES", seg, np.repeat(cols, 2, axis=0))
        B["origins"] = C.make_batch("uniform", "POINTS", O[mask])

    if settings.show_hits:
        hm = res["hit"] & mask
        if hm.any():
            if settings.color_by_object:
                cols = np.array([_object_color(k) for k in res["hobj"][hm]])
            else:
                cols = np.array([STATUS_COLORS[s] for s in status[hm]])
            B["hits"] = C.make_batch("smooth", "POINTS", res["hp"][hm], cols)
    return B


# ---------------------------------------------------------------------------
# Refresh / overlay management
# ---------------------------------------------------------------------------

def _signature(scene, view_layer, settings):
    bake = scene.render.bake
    act = view_layer.objects.active
    sel = sorted(o.name for o in view_layer.objects if o.select_get())
    mats = tuple(tuple(round(x, 6) for r in view_layer.objects[n].matrix_world for x in r)
                 for n in sel if n in view_layer.objects)
    return (
        act.name if act else "", tuple(sel), mats, act.mode if act else "",
        round(bake.cage_extrusion, 7), round(bake.max_ray_distance, 7), bake.use_cage,
        bake.cage_object.name if bake.cage_object else "", bake.use_selected_to_active,
        settings.samples, settings.only_selected,
    )


def refresh(context=None, force=False):
    ctx = context or bpy.context
    scene = ctx.scene
    view_layer = ctx.view_layer
    s = scene.tk_bake
    sig = _signature(scene, view_layer, s)
    if not force and not _state["dirty"] and sig == _state["sig"]:
        return
    _state["sig"] = sig
    _state["dirty"] = False
    try:
        dg = ctx.evaluated_depsgraph_get()
        res, err = compute(scene, view_layer, dg, s)
    except Exception as ex:  # keep the overlay alive on unexpected data
        res, err = None, f"Ошибка: {ex}"
    _state["result"] = res
    _state["error"] = err
    _state["batches"] = build_batches(res, s)
    C.tag_redraw_3d()


def _timer():
    try:
        scene = bpy.context.scene
        s = scene.tk_bake
    except Exception:
        _state["timer"] = False
        return None
    if not s.enabled:
        _state["timer"] = False
        return None
    if s.auto_update:
        refresh()
    return 0.3


@bpy.app.handlers.persistent
def _on_depsgraph(scene, depsgraph):
    if not getattr(scene, "tk_bake", None) or not scene.tk_bake.enabled:
        return
    for u in depsgraph.updates:
        idb = u.id
        if isinstance(idb, bpy.types.Object) and u.is_updated_geometry:
            name = idb.original.name if hasattr(idb, "original") else idb.name
            _state["geo_ver"][name] = _state["geo_ver"].get(name, 0) + 1
            _state["dirty"] = True
        elif isinstance(idb, bpy.types.Scene):
            _state["dirty"] = True


@bpy.app.handlers.persistent
def _on_load(*_a):
    disable_overlay(reset_prop=False)
    _state["bvh"].clear()
    _state["geo_ver"].clear()
    try:
        if bpy.context.scene.tk_bake.enabled:
            enable_overlay()
    except Exception:
        pass


def _draw_3d():
    B = _state["batches"]
    if not B:
        return
    try:
        s = bpy.context.scene.tk_bake
    except Exception:
        return
    with C.GPUState(depth=True):
        C.draw_batch(B.get("heat"))
        C.draw_batch(B.get("shell_tris"), (0.35, 0.75, 1.0, s.shell_alpha))
        C.draw_batch(B.get("shell_edges"), (0.35, 0.8, 1.0, 0.85), 1.4)
        C.draw_batch(B.get("limit_edges"), (0.65, 0.4, 1.0, 0.85), 1.2)
        C.draw_batch(B.get("rays"), None, 1.2)
        C.draw_batch(B.get("origins"), (0.6, 0.85, 1.0, 0.7), point_size=3.0)
        C.draw_batch(B.get("hits"), None, point_size=4.5)
    probe = _state.get("probe")
    if probe and probe.get("batches"):
        with C.GPUState(depth=False):
            pb = probe["batches"]
            C.draw_batch(pb.get("ray"), None, 3.0)
            C.draw_batch(pb.get("pts"), None, point_size=9.0)


def _draw_2d():
    try:
        s = bpy.context.scene.tk_bake
    except Exception:
        return
    region = bpy.context.region
    probe = _state.get("probe")
    if probe and probe.get("text") and probe.get("mouse"):
        mx, my = probe["mouse"]
        C.draw_text_lines(probe["text"], mx + 22, my + 10, size=13)
    if not s.show_hud:
        return
    lines = hud_lines()
    if lines and region:
        C.draw_text_lines(lines, 18, region.height - 70, size=12, line_h=17)


def hud_lines():
    if _state["error"]:
        return [("Запекание: " + _state["error"], (1.0, 0.7, 0.3, 1.0))]
    res = _state["result"]
    if not res:
        return []
    st = res["stats"]
    n = max(1, st["n"])
    white = (0.95, 0.95, 0.95, 1.0)
    lines = [
        (f"Запекание: {', '.join(st['highs'])} → {st['low']}", white),
        (("Cage: " + st["cage"]) if st["cage"] else
         f"Extrusion {C.fmt_len(st['extrusion'])} · Max Ray "
         f"{C.fmt_len(st['max_dist']) if st['max_dist'] > 0 else '∞'}", white),
        (f"Попало {st['ok'] * 100 / n:.1f}%   промахи {st['miss']}   изнанка {st['back']}"
         f"   подозрительно {st['warn']}", white),
    ]
    for h in st["hints"][:3]:
        lines.append(("⚠ " + h, (1.0, 0.8, 0.35, 1.0)))
    return lines


def enable_overlay():
    if C.can_draw():
        if _state["h3d"] is None:
            _state["h3d"] = bpy.types.SpaceView3D.draw_handler_add(
                _draw_3d, (), "WINDOW", "POST_VIEW")
        if _state["h2d"] is None:
            _state["h2d"] = bpy.types.SpaceView3D.draw_handler_add(
                _draw_2d, (), "WINDOW", "POST_PIXEL")
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    _state["dirty"] = True
    if not _state["timer"]:
        _state["timer"] = True
        bpy.app.timers.register(_timer, first_interval=0.05)


def disable_overlay(reset_prop=False):
    for key in ("h3d", "h2d"):
        if _state[key] is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(_state[key], "WINDOW")
            except Exception:
                pass
            _state[key] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    _state["batches"] = {}
    _state["result"] = None
    C.tag_redraw_3d() if bpy.context.window_manager else None


# ---------------------------------------------------------------------------
# Auto-fit
# ---------------------------------------------------------------------------

def autofit(scene, view_layer, depsgraph, samples=6000, percentile=99.0, margin=1.15):
    """Measure how far the high-poly lies outside/inside the low-poly along the bake
    normal and return (extrusion, max_ray_distance, unmatched_fraction)."""
    low, highs, err = gather(scene, view_layer)
    if err:
        raise RuntimeError(err)
    L = C.mesh_arrays(low, depsgraph)
    co_w, vn_w, pn_w = L.world_co(), L.world_vn(), L.world_pn()
    diag = C.bbox_diag(co_w)
    rng = np.random.default_rng(11)
    ids, bary = _sample_tris(L, co_w, samples, None, rng)
    P, N, _, _ = ray_setup(L, co_w, vn_w, pn_w, ids, bary, 0.0, None)
    hb = [_bvh_for(h, depsgraph) for h in highs]
    need_out, need_in = [], []
    unmatched = 0
    for i in range(len(P)):
        p = Vector(P[i])
        n = Vector(N[i])
        best_out = best_in = None
        for (bvh, m, minv, minv3) in hb:
            r = C.world_ray_bvh(bvh, m, minv, minv3, p, n, None)
            if r and (best_out is None or r[2] < best_out):
                best_out = r[2]
            r = C.world_ray_bvh(bvh, m, minv, minv3, p, -n, None)
            if r and (best_in is None or r[2] < best_in):
                best_in = r[2]
        if best_out is None and best_in is None:
            unmatched += 1
        elif best_in is None or (best_out is not None and best_out <= best_in):
            need_out.append(best_out)
        else:
            need_in.append(best_in)
    small = diag * 1e-4
    po = float(np.percentile(need_out, percentile)) if need_out else 0.0
    pi = float(np.percentile(need_in, percentile)) if need_in else 0.0
    extrusion = po * margin + small
    max_dist = extrusion + pi * margin + small
    return extrusion, max_dist, unmatched / max(1, len(P))


# ---------------------------------------------------------------------------
# Probe (hover a point on the low-poly and see its ray)
# ---------------------------------------------------------------------------

def probe_at(context, region, rv3d, coord):
    from bpy_extras import view3d_utils
    res = _state["result"]
    if not res:
        return None
    scene = context.scene
    vl = context.view_layer
    low = vl.objects.active
    if low is None or low.name != res["low"].name:
        return None
    dg = context.evaluated_depsgraph_get()
    lb = _bvh_for(low, dg)
    ro = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    rd = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    bvh, m, minv, minv3 = lb
    lo = minv @ ro
    ld = (minv3 @ rd).normalized()
    hit = bvh.ray_cast(lo, ld)
    if hit[0] is None:
        return None
    poly = hit[2]
    L = res["low"]
    if poly is None or poly >= len(L.loop_start):
        return None
    ls, lt = L.loop_start[poly], L.loop_total[poly]
    pv = L.loop_vert[ls:ls + lt]
    co_w, vn_w, pn_w = res["co_w"], res["vn_w"], res["pn_w"]
    Pw = m @ hit[0]
    w = poly_3d_calc([Vector(co_w[v]) for v in pv], Pw)
    if L.poly_smooth[poly]:
        N = Vector(np.sum(vn_w[pv] * np.array(w)[:, None], axis=0)).normalized()
    else:
        N = Vector(pn_w[poly])
    e = res["extrusion"]
    limit = res["limit"]
    if res["cage_w"] is not None:
        O = Vector(np.sum(res["cage_w"][pv] * np.array(w)[:, None], axis=0))
        D = (Pw - O)
        D = D.normalized() if D.length > 1e-9 else -N
    else:
        O = Pw + N * e
        D = -N
    best = None
    bname = ""
    for k, hb in enumerate(res["highs_bvh"]):
        r = C.world_ray_bvh(*hb, O, D, limit)
        if r and (best is None or r[2] < best[2]):
            best = r
            bname = res["high_names"][k]
    out = {"P": Pw, "N": N, "O": O, "D": D, "hit": best, "name": bname, "limit": limit, "e": e}
    return out


def probe_batches(p, diag):
    B = {}
    if not C.can_draw() or not p:
        return B
    O, D = p["O"], p["D"]
    if p["hit"]:
        end = p["hit"][0]
        col = STATUS_COLORS[STATUS_BACK if p["hit"][1].dot(p["hit"][3]) > 0 else STATUS_OK]
    else:
        vis = p["limit"] if p["limit"] else max(2 * p["e"], 0.03 * diag)
        end = O + D * vis
        col = STATUS_COLORS[STATUS_MISS]
    B["ray"] = C.make_batch("smooth", "LINES", [O, end], [col, col])
    pts = [O, p["P"]] + ([p["hit"][0]] if p["hit"] else [])
    cols = [(0.4, 0.8, 1, 1), (1, 1, 1, 1)] + ([col] if p["hit"] else [])
    B["pts"] = C.make_batch("smooth", "POINTS", pts, cols)
    return B


def probe_text(p):
    if not p:
        return [("Наведите на low-poly", (1, 1, 1, 0.9))]
    white = (0.95, 0.95, 0.95, 1)
    lines = []
    if p["hit"]:
        hitp, ln, dist, ld = p["hit"]
        depth = (hitp - p["P"]).dot(p["N"])
        side = "снаружи" if depth >= 0 else "внутри"
        lines.append((f"→ {p['name']}", (0.5, 1.0, 0.6, 1)))
        lines.append((f"high {side} на {C.fmt_len(abs(depth))}", white))
        lines.append((f"длина луча {C.fmt_len(dist)}", white))
        if p["limit"]:
            lines.append((f"запас до Max Ray: {C.fmt_len(p['limit'] - dist)}", white))
        if depth >= 0 and p["e"] > 0:
            lines.append((f"запас Extrusion: {C.fmt_len(p['e'] - depth)}", white))
        if ln.dot(ld) > 0:
            lines.append(("попадание в изнанку!", STATUS_COLORS[STATUS_BACK]))
    else:
        lines.append(("промах — сюда ничего не запечётся", STATUS_COLORS[STATUS_MISS]))
        lines.append((f"Extrusion {C.fmt_len(p['e'])}, Max Ray "
                      f"{C.fmt_len(p['limit']) if p['limit'] else '∞'}", white))
    return lines


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class TK_OT_bake_preview_refresh(bpy.types.Operator):
    bl_idname = "tk.bake_preview_refresh"
    bl_label = "Обновить просмотр"
    bl_description = "Пересчитать лучи запекания"

    def execute(self, context):
        _state["bvh"].clear()
        refresh(context, force=True)
        if _state["error"]:
            self.report({"WARNING"}, _state["error"])
        return {"FINISHED"}


class TK_OT_bake_autofit(bpy.types.Operator):
    bl_idname = "tk.bake_autofit"
    bl_label = "Подобрать Extrusion и Max Ray"
    bl_description = ("Измерить, насколько high-poly выходит наружу и уходит внутрь low-poly, "
                      "и выставить Extrusion / Max Ray Distance с запасом")
    bl_options = {"REGISTER", "UNDO"}

    percentile: FloatProperty(name="Покрыть точек, %", default=99.0, min=50.0, max=100.0)
    margin: FloatProperty(name="Запас", default=1.15, min=1.0, max=3.0)
    samples: IntProperty(name="Точек", default=6000, min=200, max=100000)

    def execute(self, context):
        try:
            e, d, unmatched = autofit(context.scene, context.view_layer,
                                      context.evaluated_depsgraph_get(),
                                      self.samples, self.percentile, self.margin)
        except RuntimeError as ex:
            self.report({"WARNING"}, str(ex))
            return {"CANCELLED"}
        bake = context.scene.render.bake
        bake.use_cage = False
        bake.cage_extrusion = e
        bake.max_ray_distance = d
        _state["dirty"] = True
        msg = f"Extrusion = {C.fmt_len(e)}, Max Ray Distance = {C.fmt_len(d)}"
        if unmatched > 0.001:
            msg += f"; {unmatched * 100:.1f}% точек не нашли high-poly ни снаружи, ни внутри"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class TK_OT_bake_select_problems(bpy.types.Operator):
    bl_idname = "tk.bake_select_problems"
    bl_label = "Выделить проблемные грани"
    bl_description = "Выделить на low-poly грани, где есть промахи, изнанка или подозрительные лучи"
    bl_options = {"REGISTER", "UNDO"}

    include_warn: BoolProperty(name="Включая подозрительные", default=True)

    def execute(self, context):
        import bmesh
        res = _state["result"]
        if not res:
            self.report({"WARNING"}, "Сначала включите просмотр запекания")
            return {"CANCELLED"}
        low = context.view_layer.objects.get(res["stats"]["low"])
        if low is None:
            return {"CANCELLED"}
        L = res["low"]
        bad = res["status"] != STATUS_OK
        if not self.include_warn:
            bad &= res["status"] != STATUS_WARN
        polys = set(np.unique(L.tri_poly[res["ids"][bad]]).tolist())
        if low.mode != "EDIT":
            context.view_layer.objects.active = low
            bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(low.data)
        bm.faces.ensure_lookup_table()
        if len(bm.faces) != len(L.loop_start):
            self.report({"WARNING"}, "Модификаторы меняют топологию — выделение может не совпасть")
        for f in bm.faces:
            f.select_set(f.index in polys)
        bm.select_flush_mode()
        context.tool_settings.mesh_select_mode = (False, False, True)
        bmesh.update_edit_mesh(low.data)
        self.report({"INFO"}, f"Выделено граней: {len(polys)}")
        return {"FINISHED"}


class TK_OT_bake_probe(bpy.types.Operator):
    bl_idname = "tk.bake_probe"
    bl_label = "Щуп запекания"
    bl_description = ("Наведите курсор на low-poly: виден луч запекания в этой точке, "
                      "куда он попадает и сколько остаётся запаса. Esc / ПКМ — выход")

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        s = context.scene.tk_bake
        if not s.enabled:
            s.enabled = True
        refresh(context, force=True)
        if not _state["result"]:
            self.report({"WARNING"}, _state["error"] or "Нечего проверять")
            return {"CANCELLED"}
        _state["probe"] = {"batches": {}, "text": None, "mouse": None}
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("Щуп запекания: двигайте курсор по low-poly · Esc / ПКМ — выход")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            _state["probe"] = None
            context.workspace.status_text_set(None)
            context.area.tag_redraw()
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            region, rv3d = context.region, context.region_data
            if region is None or rv3d is None:
                return {"PASS_THROUGH"}
            coord = (event.mouse_region_x, event.mouse_region_y)
            try:
                p = probe_at(context, region, rv3d, coord)
            except Exception:
                p = None
            diag = _state["result"]["diag"] if _state["result"] else 1.0
            _state["probe"] = {"batches": probe_batches(p, diag), "text": probe_text(p),
                               "mouse": coord}
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


class TK_OT_bake_toggle(bpy.types.Operator):
    bl_idname = "tk.bake_preview_toggle"
    bl_label = "Просмотр запекания"
    bl_description = "Включить / выключить просмотр лучей запекания во вьюпорте"

    def execute(self, context):
        s = context.scene.tk_bake
        s.enabled = not s.enabled
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def draw_bake_ui(layout, context):
    s = context.scene.tk_bake
    bake = context.scene.render.bake
    row = layout.row(align=True)
    row.scale_y = 1.3
    row.prop(s, "enabled", text="Просмотр запекания", toggle=True,
             icon="HIDE_OFF" if s.enabled else "HIDE_ON")
    row.operator("tk.bake_preview_refresh", text="", icon="FILE_REFRESH")

    col = layout.column(align=True)
    col.prop(bake, "use_cage", text="Cage")
    if bake.use_cage:
        col.prop(bake, "cage_object", text="")
    col.prop(bake, "cage_extrusion", text="Extrusion")
    col.prop(bake, "max_ray_distance", text="Max Ray Distance")
    layout.operator("tk.bake_autofit", icon="DRIVER_DISTANCE")
    row = layout.row(align=True)
    row.operator("tk.bake_probe", icon="EYEDROPPER")
    row.operator("tk.bake_select_problems", text="Проблемы", icon="RESTRICT_SELECT_OFF")

    if s.enabled:
        res = _state["result"]
        if _state["error"]:
            layout.label(text=_state["error"], icon="ERROR")
        elif res:
            st = res["stats"]
            n = max(1, st["n"])
            box = layout.box()
            box.label(text=f"Попало: {st['ok'] * 100 / n:.1f}%  ({st['ok']}/{st['n']})",
                      icon="CHECKMARK")
            r = box.row()
            r.label(text=f"Промахи: {st['miss']}", icon="CANCEL")
            r.label(text=f"Изнанка: {st['back']}")
            box.label(text=f"Подозрительные: {st['warn']}  ·  {st['time'] * 1000:.0f} мс",
                      icon="QUESTION")
            for name, cnt in st["per_obj"].items():
                box.label(text=f"{name}: {cnt}", icon="OBJECT_DATA")
            for h in st["hints"]:
                box.label(text=h, icon="INFO")

    sub = layout.column(align=True)
    sub.label(text="Показывать:")
    sub.prop(s, "show_shell")
    sub.prop(s, "show_limit")
    sub.prop(s, "show_rays")
    sub.prop(s, "show_hits")
    sub.prop(s, "show_heat")
    sub.prop(s, "show_hud")
    sub.prop(s, "color_by_object")
    sub.prop(s, "only_problems")
    sub.prop(s, "only_selected")
    layout.prop(s, "samples")
    layout.prop(s, "shell_alpha", slider=True)
    layout.prop(s, "auto_update")


class TK_PT_bake_preview(bpy.types.Panel):
    bl_label = "Просмотр запекания"
    bl_idname = "TK_PT_bake_preview"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_order = 0

    def draw(self, context):
        self.layout.label(text="Выделите high-poly, затем low-poly", icon="INFO")
        draw_bake_ui(self.layout, context)


def _render_bake_append(self, context):
    layout = self.layout
    s = context.scene.tk_bake
    row = layout.row(align=True)
    row.prop(s, "enabled", text="Просмотр лучей во вьюпорте", toggle=True, icon="HIDE_OFF")
    row.operator("tk.bake_autofit", text="", icon="DRIVER_DISTANCE")


classes = (
    TK_BakePreviewSettings,
    TK_OT_bake_preview_refresh,
    TK_OT_bake_autofit,
    TK_OT_bake_select_problems,
    TK_OT_bake_probe,
    TK_OT_bake_toggle,
    TK_PT_bake_preview,
)

_appended = []


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_bake = bpy.props.PointerProperty(type=TK_BakePreviewSettings)
    bpy.app.handlers.load_post.append(_on_load)
    for pname in ("CYCLES_RENDER_PT_bake_selected_to_active", "CYCLES_RENDER_PT_bake"):
        pt = getattr(bpy.types, pname, None)
        if pt is not None:
            pt.append(_render_bake_append)
            _appended.append(pt)
            break


def unregister():
    disable_overlay()
    for pt in _appended:
        try:
            pt.remove(_render_bake_append)
        except Exception:
            pass
    _appended.clear()
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    del bpy.types.Scene.tk_bake
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
