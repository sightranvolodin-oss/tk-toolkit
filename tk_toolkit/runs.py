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
    __slots__ = ("idx", "center", "angle", "length", "zmin", "zmax", "pts", "line", "run")


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
    for k, ln in enumerate(lines):
        for m in ln["parts"]:
            m.line = k
    return lines


def attach_small(parts, lines, dist=0.15, margin=0.5):
    left = []
    for p in parts:
        if p.line >= 0:
            continue
        best, bd = -1, dist
        for k, ln in enumerate(lines):
            u = p.center @ ln["dir"]
            if u < ln["u0"] - margin or u > ln["u1"] + margin:
                continue
            dd = abs(p.center @ ln["nrm"] - ln["offset"])
            if dd < bd:
                best, bd = k, dd
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


def analyse_mesh(W, me):
    lab, n = mesh_loose_parts(me)
    parts = describe_parts(W, lab, n)
    lines = find_lines(parts)
    left = attach_small(parts, lines)
    runs = split_runs(lines)
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
