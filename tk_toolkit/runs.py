"""Split a big railing mesh into straight runs (pure numpy, no scipy).

  1. loose parts (connected components of the mesh);
  2. elongated parts (handrails, rails, panels) define straight lines in plan;
     small parts (balusters, posts) are attached to the nearest line;
  3. along each line, parts that touch / nearly touch in (u, z) form one run —
     so a second railing on the same line but at another level is its own run.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def components(n, a, b):
    """Labels 0..k-1 for *n* nodes joined by edges (a[i], b[i]); vectorised union-find."""
    parent = np.arange(n, dtype=np.int64)
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    while True:
        pa, pb = parent[a], parent[b]
        diff = pa != pb
        if not diff.any():
            break
        lo = np.minimum(pa[diff], pb[diff])
        hi = np.maximum(pa[diff], pb[diff])
        np.minimum.at(parent, hi, lo)            # hook bigger root under smaller
        while True:                               # pointer jumping
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                break
            parent = nxt
    _, lab = np.unique(parent, return_inverse=True)
    return lab


def mesh_loose_parts(me):
    """(vertex_labels, n_parts) of a bpy Mesh."""
    nv = len(me.vertices)
    ne = len(me.edges)
    e = np.empty(ne * 2, np.int64)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    lab = components(nv, e[:, 0], e[:, 1])
    return lab, int(lab.max()) + 1 if nv else 0


# ---------------------------------------------------------------------------
# Parts → lines → runs
# ---------------------------------------------------------------------------

class Part:
    __slots__ = ("idx", "center", "angle", "length", "zmin", "zmax", "pts", "line", "run",
                 "mat")


def describe_parts(W, lab, n):
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    starts = np.searchsorted(ls, np.arange(n))
    ends = np.searchsorted(ls, np.arange(n), side="right")
    parts = []
    for i in range(n):
        P = W[order[starts[i]:ends[i]]]
        p = Part()
        p.idx = i
        p.pts = P
        xy = P[:, :2]
        p.center = xy.mean(axis=0)
        p.zmin, p.zmax = float(P[:, 2].min()), float(P[:, 2].max())
        p.angle = None
        p.length = 0.0
        p.line = -1
        p.run = -1
        p.mat = 0
        if len(P) >= 3:
            q = xy - p.center
            w, v = np.linalg.eigh(q.T @ q)
            d = v[:, 1]
            L = float(np.ptp(q @ d))
            wd = float(np.ptp(q @ v[:, 0]))
            if L > 0.3 and L > 3 * max(wd, 0.01):
                p.angle = math.degrees(math.atan2(d[1], d[0])) % 180.0
                p.length = L
        parts.append(p)
    return parts


def _group_sorted(values, gap):
    """Indices grouped where sorted values differ by more than *gap*."""
    order = np.argsort(values)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] > gap:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    return groups


def find_lines(parts, angle_tol=3.0, offset_tol=0.15):
    """Collinear long parts → lines. Parts of different materials never share a line:
    a red stair railing and a white fence standing next to each other stay apart."""
    lines = []
    for mat in sorted({p.mat for p in parts}):
        lines += _find_lines_one([p for p in parts if p.mat == mat], angle_tol, offset_tol)
    for k, ln in enumerate(lines):
        ln["mat"] = ln["parts"][0].mat
        for m in ln["parts"]:
            m.line = k
    return lines


def _find_lines_one(parts, angle_tol, offset_tol):
    el = [p for p in parts if p.angle is not None]
    if not el:
        return []
    ang = np.array([p.angle for p in el])
    ang = np.where(ang > 180 - angle_tol, ang - 180, ang)       # 179° ≈ -1°
    lines = []
    for g in _group_sorted(ang, angle_tol):
        theta = math.radians(float(np.average(ang[g], weights=[el[i].length for i in g])))
        d = np.array([math.cos(theta), math.sin(theta)])
        nrm = np.array([-d[1], d[0]])
        off = np.array([el[i].center @ nrm for i in g])
        for h in _group_sorted(off, offset_tol):
            members = [el[g[k]] for k in h]
            o = float(np.average(off[h], weights=[m.length for m in members]))
            us = np.concatenate([m.pts[:, :2] @ d for m in members])
            lines.append({"dir": d, "nrm": nrm, "offset": o,
                          "u0": float(us.min()), "u1": float(us.max()), "parts": members})
    return lines


def attach_small(parts, lines, dist=0.15, margin=0.5):
    left = []
    for p in parts:
        if p.line >= 0:
            continue
        # a line of the part's own material wins; otherwise any (a railing may well be
        # posts of one material and rails of another)
        best, bd, best_same = -1, dist, False
        for k, ln in enumerate(lines):
            u = p.center @ ln["dir"]
            if u < ln["u0"] - margin or u > ln["u1"] + margin:
                continue
            dd = abs(p.center @ ln["nrm"] - ln["offset"])
            same = ln.get("mat", p.mat) == p.mat
            if dd < dist and ((same and not best_same) or (same == best_same and dd < bd)):
                best, bd, best_same = k, dd, same
        if best >= 0:
            p.line = best
            lines[best]["parts"].append(p)
        else:
            left.append(p)
    return left


def split_runs(lines, gap=0.15):
    """Within each line, parts that really come closer than ~*gap* form a run.

    Closeness is measured on the points themselves (a grid of *gap*-sized cells in
    (u, z)), not on bounding boxes: the box of a sloped handrail is a big rectangle that
    would glue together railings of two different floors."""
    runs = []
    for k, ln in enumerate(lines):
        ps = ln["parts"]
        n = len(ps)
        if n == 0:
            continue
        cu, cz, owner = [], [], []
        for i, p in enumerate(ps):
            u = p.pts[:, :2] @ ln["dir"]
            z = p.pts[:, 2]
            # low-poly parts (a 40 m box rail has vertices only at its ends): fill the cells
            # between points that are far apart along u, so the part counts as solid
            o = np.argsort(u, kind="stable")
            us, zs = u[o], z[o]
            du = np.diff(us)
            far = np.nonzero(du > gap)[0]
            if len(far):
                fu, fz = [us], [zs]
                for j in far[:2000]:
                    k = int(np.ceil(du[j] / (0.5 * gap)))
                    t = np.linspace(0.0, 1.0, k + 1)[1:-1]
                    fu.append(us[j] + t * du[j])
                    fz.append(zs[j] + t * (zs[j + 1] - zs[j]))
                us, zs = np.concatenate(fu), np.concatenate(fz)
            cells = np.unique(np.stack([np.floor(us / gap), np.floor(zs / gap)], 1)
                              .astype(np.int64), axis=0)
            cu.append(cells[:, 0])
            cz.append(cells[:, 1])
            owner.append(np.full(len(cells), i, np.int64))
        cu, cz, owner = np.concatenate(cu), np.concatenate(cz), np.concatenate(owner)
        a, b = [], []
        base_u, base_z = cu.min() - 2, cz.min() - 2
        span = int(cz.max() - base_z + 3)
        for du in (-1, 0, 1):                     # touching cells, 8-neighbourhood
            for dz in (-1, 0, 1):
                key = (cu + du - base_u) * span + (cz + dz - base_z)
                key0 = (cu - base_u) * span + (cz - base_z)
                # owners of key0 vs owners of shifted key
                order = np.argsort(key0, kind="stable")
                k0s = key0[order]
                pos = np.searchsorted(k0s, key)
                hit = (pos < len(k0s)) & (k0s[np.minimum(pos, len(k0s) - 1)] == key)
                other = owner[order[np.minimum(pos, len(k0s) - 1)]]
                m = hit & (other != owner)
                a.append(owner[m])
                b.append(other[m])
        a = np.concatenate(a) if a else np.zeros(0, np.int64)
        b = np.concatenate(b) if b else np.zeros(0, np.int64)
        lab = components(n, a, b)
        for r in range(int(lab.max()) + 1):
            members = [ps[i] for i in range(n) if lab[i] == r]
            run = {"line": k, "dir": ln["dir"], "nrm": ln["nrm"], "parts": members}
            us = np.concatenate([m.pts[:, :2] @ ln["dir"] for m in members])
            run["u0"], run["u1"] = float(us.min()), float(us.max())
            run["z0"] = min(m.zmin for m in members)
            run["z1"] = max(m.zmax for m in members)
            runs.append(run)
    for i, r in enumerate(runs):
        for m in r["parts"]:
            m.run = i
    return runs


def bar_rows(left, runs, step_max=0.35, min_bars=3):
    """Rows of loose vertical bars with no rail of their own (a grille, a screen): they
    are not on any line, so group them by plan position — neighbours closer than
    *step_max*, at the same height, on one straight line → a run of their own."""
    vert = [p for p in left
            if (p.zmax - p.zmin) > 3 * max(np.ptp(p.pts[:, 0]), np.ptp(p.pts[:, 1]), 1e-6)]
    if len(vert) < min_bars:
        return left
    C = np.array([[p.center[0], p.center[1], 0.5 * (p.zmin + p.zmax)] for p in vert])
    n = len(vert)
    a, b = [], []
    for i in range(n):                                  # small n (leftovers only)
        d = np.linalg.norm(C[:, :2] - C[i, :2], axis=1)
        # bars of a sloped flight stand at different heights: allow the slope
        m = (d < step_max) & (np.abs(C[:, 2] - C[i, 2]) < 0.3 + 1.0 * d) & (np.arange(n) > i)
        js = np.nonzero(m)[0]
        a += [i] * len(js)
        b += js.tolist()
    lab = components(n, np.array(a, np.int64), np.array(b, np.int64))
    used = set()
    for r in range(int(lab.max()) + 1 if n else 0):
        idx = np.nonzero(lab == r)[0]
        if len(idx) < min_bars:
            continue
        P = C[idx, :2]
        c = P.mean(0)
        _, sv, vt = np.linalg.svd(P - c)
        d_ = vt[0]
        if sv[0] < 1e-6 or (len(sv) > 1 and sv[1] > 0.05 * sv[0] + 0.02):
            continue                                    # not a straight row
        nrm = np.array([-d_[1], d_[0]])
        members = [vert[i] for i in idx]
        us = np.concatenate([m.pts[:, :2] @ d_ for m in members])
        run = {"line": -1, "dir": d_, "nrm": nrm, "parts": members,
               "u0": float(us.min()), "u1": float(us.max()),
               "z0": min(m.zmin for m in members), "z1": max(m.zmax for m in members)}
        k = len(runs)
        runs.append(run)
        for m in members:
            m.run = k
            used.add(id(m))
    return [p for p in left if id(p) not in used]


def part_materials(me, lab, n):
    """Dominant material index of every loose part."""
    npo = len(me.polygons)
    if npo == 0 or len(me.materials) <= 1:
        return np.zeros(n, np.int64)
    ls = np.empty(npo, np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lv = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lv)
    mi = np.empty(npo, np.int64)
    me.polygons.foreach_get("material_index", mi)
    plab = lab[lv[ls]]
    nm = int(mi.max()) + 1
    cnt = np.bincount(plab * nm + mi, minlength=n * nm).reshape(n, nm)
    return cnt.argmax(1)


def merge_tiny(runs, min_len=0.25, gap=0.35):
    """A corner or end post often comes out as a 'run' of its own (4 cm long): give it to
    the neighbouring run in the same plane instead of throwing it away."""
    def off(r):
        c = np.concatenate([p.pts[:, :2] for p in r["parts"]])
        return float(np.median(c @ np.asarray(r["nrm"])))
    big = [k for k, r in enumerate(runs) if r["parts"] and r["u1"] - r["u0"] >= min_len]
    offs = {k: off(runs[k]) for k in big}
    for t, r in enumerate(runs):
        if not r["parts"] or r["u1"] - r["u0"] >= min_len:
            continue
        P = np.concatenate([p.pts for p in r["parts"]])
        best, bd = -1, gap
        for k in big:
            q = runs[k]
            d2, n2 = np.asarray(q["dir"]), np.asarray(q["nrm"])
            if abs(float(np.mean(P[:, :2] @ n2)) - offs[k]) > 0.15:
                continue
            u = P[:, :2] @ d2
            ug = max(0.0, q["u0"] - u.max(), u.min() - q["u1"])
            if ug >= bd:
                continue
            # height of the neighbour right where the post stands (a sloped run's bounding
            # box spans the floors above and below)
            Q = np.concatenate([p.pts for p in q["parts"]])
            qu = Q[:, :2] @ d2
            near = np.abs(qu - float(np.median(u))) < 0.5
            if not near.any():
                continue
            if P[:, 2].max() < Q[near, 2].min() - 0.1 or P[:, 2].min() > Q[near, 2].max() + 0.1:
                continue
            best, bd = k, ug
        if best < 0:
            continue
        q = runs[best]
        u = P[:, :2] @ np.asarray(q["dir"])
        q["parts"] = q["parts"] + r["parts"]
        q["u0"], q["u1"] = min(q["u0"], float(u.min())), max(q["u1"], float(u.max()))
        q["z0"], q["z1"] = min(q["z0"], float(P[:, 2].min())), max(q["z1"], float(P[:, 2].max()))
        for p in r["parts"]:
            p.run = best
        r["parts"] = []
        r["u1"] = r["u0"]


def analyse_mesh(W, me):
    lab, n = mesh_loose_parts(me)
    parts = describe_parts(W, lab, n)
    pm = part_materials(me, lab, n)
    for p in parts:
        p.mat = int(pm[p.idx])
    lines = find_lines(parts)
    left = attach_small(parts, lines)
    runs = split_runs(lines)
    left = bar_rows(left, runs)
    merge_tiny(runs)
    return lab, parts, lines, runs, left


# ---------------------------------------------------------------------------
# Sub-mesh of selected polygons (fast, foreach based)
# ---------------------------------------------------------------------------

def mesh_cache(me):
    """Arrays of *me* read once and shared by many submesh() calls."""
    npoly, nl, nv = len(me.polygons), len(me.loops), len(me.vertices)
    c = {}
    c["ls"] = np.empty(npoly, np.int64); me.polygons.foreach_get("loop_start", c["ls"])
    c["lt"] = np.empty(npoly, np.int64); me.polygons.foreach_get("loop_total", c["lt"])
    c["lv"] = np.empty(nl, np.int64); me.loops.foreach_get("vertex_index", c["lv"])
    co = np.empty(nv * 3, np.float32); me.vertices.foreach_get("co", co)
    c["co"] = co.reshape(-1, 3)
    c["mi"] = np.empty(npoly, np.int32); me.polygons.foreach_get("material_index", c["mi"])
    c["sm"] = np.empty(npoly, bool); me.polygons.foreach_get("use_smooth", c["sm"])
    if me.uv_layers:
        uv = np.empty(nl * 2, np.float32)
        me.uv_layers[0].data.foreach_get("uv", uv)
        c["uv"] = uv.reshape(-1, 2)
        c["uvname"] = me.uv_layers[0].name
    return c


def submesh(me, poly_mask, name, cache=None):
    """New bpy Mesh made of the polygons where *poly_mask* is True (keeps materials, UVs
    of the first layer and smooth flags)."""
    import bpy
    c = cache if cache is not None else mesh_cache(me)
    ls, lt, lv = c["ls"], c["lt"], c["lv"]
    pidx = np.nonzero(poly_mask)[0]
    tot = lt[pidx]
    if len(pidx):
        starts_new = np.concatenate([[0], np.cumsum(tot)[:-1]])
        loops = np.repeat(ls[pidx] - starts_new, tot) + np.arange(int(tot.sum()))
    else:
        loops = np.zeros(0, np.int64)
    used_v = np.unique(lv[loops])
    remap = np.full(len(c["co"]), -1, np.int64)
    remap[used_v] = np.arange(len(used_v))
    co = c["co"][used_v]
    new = bpy.data.meshes.new(name)
    new.vertices.add(len(used_v))
    new.vertices.foreach_set("co", co.ravel())
    new.loops.add(len(loops))
    new.loops.foreach_set("vertex_index", remap[lv[loops]].astype(np.int32))
    new.polygons.add(len(pidx))
    starts = np.concatenate([[0], np.cumsum(tot)[:-1]]) if len(pidx) else np.zeros(0, np.int64)
    new.polygons.foreach_set("loop_start", starts.astype(np.int32))
    if hasattr(new.polygons[0] if len(pidx) else None, "loop_total"):
        try:
            new.polygons.foreach_set("loop_total", tot.astype(np.int32))
        except (AttributeError, RuntimeError, TypeError):
            pass
    new.polygons.foreach_set("material_index", c["mi"][pidx])
    new.polygons.foreach_set("use_smooth", c["sm"][pidx])
    for m in me.materials:
        new.materials.append(m)
    if "uv" in c:
        nl = new.uv_layers.new(name=c["uvname"])
        nl.data.foreach_set("uv", c["uv"][loops].ravel())
    new.update(calc_edges=True)
    new.validate(verbose=False)
    return new
