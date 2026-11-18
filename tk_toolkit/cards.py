"""Geometry → card: bake a big mesh (fence, railing, grate…) onto a plane with
transparency, tile it seamlessly and replace the geometry with that plane.

Steps
  1. orientation: vertical card (Z up) along the main horizontal direction of the
     geometry (PCA of XY), facing the chosen side;
  2. period: the repeating step (fence span) is found by autocorrelation of an
     area-weighted 2D profile of the surface along the card (or set by hand);
  3. texel density: an integer number of spans must fill the width of the texture area
     exactly, with the density inside the range of the model type
       ВПМ — 512…1702 px/m, whole tile;
       НПМ — 10…40 px/m, one quarter of the tile chosen by the user;
  4. one segment (k spans) is baked selected-to-active onto the area of the atlas
     (Diffuse, Opacity, optionally Normal / Roughness / AO);
  5. the geometry is replaced by a plane over its whole length whose UVs repeat the
     segment (one quad with wrapping on a whole tile, a strip of quads in a quarter).
"""

import math
import os

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from mathutils import Vector

from . import common as C
from . import maps as M

HIGH_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}

TYPE_DEFAULTS = {"VPM": (512.0, 1702.0), "NPM": (10.0, 40.0)}
QUADRANTS = {  # (col, row) in halves, row 0 = bottom
    "LT": (0, 1), "RT": (1, 1), "LB": (0, 0), "RB": (1, 0),
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _type_changed(self, context):
    self.density_min, self.density_max = TYPE_DEFAULTS[self.model_type]


class TK_CardSettings(bpy.types.PropertyGroup):
    model_type: EnumProperty(name="Тип модели", items=[
        ("VPM", "ВПМ", "Высокополигональная / ближний план: 512–1702 px/м, весь тайл"),
        ("NPM", "НПМ", "Низкополигональная / дальний план: 10–40 px/м, четверть тайла"),
    ], default="VPM", update=_type_changed)
    tile_size: EnumProperty(name="Тайл", items=[("2048", "2048", ""), ("4096", "4096", "")],
                            default="2048")
    density_min: FloatProperty(name="Плотность от, px/м", default=512.0, min=0.1)
    density_max: FloatProperty(name="до, px/м", default=1702.0, min=0.1)
    quadrant: EnumProperty(name="Четверть", items=[
        ("LT", "Левая верхняя", ""), ("RT", "Правая верхняя", ""),
        ("LB", "Левая нижняя", ""), ("RB", "Правая нижняя", "")], default="LT")

    follow_profile: BoolProperty(
        name="По профилю (перила на лестнице)", default=True,
        description="Если верх объекта идёт то вверх, то ровно (перила вдоль лестницы) — "
                    "выпрямить его для запекания, а карточку изогнуть по профилю. "
                    "Стойки остаются вертикальными")
    period_mode: EnumProperty(name="Пролёт", items=[
        ("AUTO", "Найти сам", "Повторяющийся шаг ищется по геометрии"),
        ("MANUAL", "Задать", ""),
        ("NONE", "Без повтора", "Запечь весь объект целиком, без тайлинга"),
    ], default="AUTO")
    period: FloatProperty(name="Пролёт", default=2.5, min=0.01, subtype="DISTANCE")
    fit_sections: BoolProperty(
        name="Подгонять шаг по участкам", default=True,
        description="На каждом марше / площадке свой шаг стоек: повтор текстуры слегка "
                    "растягивается под него, и стойки встают на свои места")
    max_tiles: bpy.props.IntProperty(
        name="Максимум тайлов", default=1, min=1, max=64,
        description="ВПМ без повтора: на сколько тайлов можно разложить объект, "
                    "чтобы удержать плотность в диапазоне")
    side: EnumProperty(name="Сторона", items=[
        ("FRONT", "Спереди", ""), ("BACK", "Сзади", "")], default="FRONT")

    ch_normal: BoolProperty(name="Normal", default=True)
    ch_roughness: BoolProperty(name="Roughness", default=False)
    ch_metallic: BoolProperty(name="Metallic", default=False)
    auto_pbr: BoolProperty(name="Metallic/Roughness — если есть в материалах", default=True,
                           description="Сам добавить каналы, если у исходных материалов "
                                       "металличность или шероховатость отличаются от обычных")
    ch_ao: BoolProperty(name="AO", default=False)
    ao_samples: FloatProperty(name="Сэмплов AO", default=32, min=1, max=1024)
    bleed: BoolProperty(name="Растянуть цвет за край", default=True)
    bleed_px: FloatProperty(name="Пикселей", default=16, min=1, max=256)
    threshold: FloatProperty(name="Порог альфы", default=0.5, min=0.0, max=1.0)
    binary_mask: BoolProperty(name="Жёсткая маска (0/1)", default=True,
                              description="Opacity строго чёрно-белая. Для НПМ порог снижается до "
                                          "0.4, чтобы тонкие стойки стали сплошным пикселем")

    atlas_name: StringProperty(name="Атлас", default="Cards",
                               description="Имя набора текстур; карточки с одним именем делят тайл")
    originals: EnumProperty(name="Исходная геометрия", items=[
        ("HIDE", "Скрыть", ""), ("DELETE", "Удалить", ""), ("KEEP", "Оставить", "")],
        default="HIDE")
    save_files: BoolProperty(name="Сохранить текстуры в папку", default=False)
    batch_rest: EnumProperty(
        name="Остатки",
        items=[("AUTO", "Авто", "НПМ — убрать, ВПМ — оставить геометрией"),
               ("KEEP", "Оставить", "То, что не стало карточками (одиночные стойки, плиты, "
                                    "кронштейны), остаётся отдельным объектом «_остаток»"),
               ("DELETE", "Убрать", "Не создавать «_остаток». Исходный меш не удаляется — "
                                    "он остаётся в сцене скрытым")],
        default="AUTO")
    output_dir: StringProperty(name="Папка", subtype="DIR_PATH", default="//textures/")
    file_format: EnumProperty(name="Формат", items=[
        ("PNG", "PNG", ""), ("TARGA", "TGA", ""), ("TIFF", "TIFF", "")], default="PNG")

    def channels(self, sources=()):
        out = ["DIFFUSE", "OPACITY"]
        if self.ch_normal:
            out.append("NORMAL")
        rough, metal = self.ch_roughness, self.ch_metallic
        if self.auto_pbr and sources:
            r_auto, m_auto = pbr_used(sources)
            rough, metal = rough or r_auto, metal or m_auto
        if rough:
            out.append("ROUGHNESS")
        if metal:
            out.append("METALLIC")
        if self.ch_ao:
            out.append("AO")
        return out


def pbr_used(objs):
    """(roughness_varies, metallic_used) over the materials of *objs*."""
    rough = metal = False
    for o in objs:
        for slot in getattr(o, "material_slots", []):
            m = slot.material
            if m is None or m.node_tree is None:
                continue
            sh = M.analyse(m)["shader"]
            if sh is None:
                continue
            mi = sh.inputs.get("Metallic")
            ri = sh.inputs.get("Roughness")
            if mi is not None and (mi.is_linked or mi.default_value > 0.01):
                metal = True
            if ri is not None and (ri.is_linked or abs(ri.default_value - 0.5) > 0.05):
                rough = True
    return rough, metal


# ---------------------------------------------------------------------------
# Analysis (pure numpy)
# ---------------------------------------------------------------------------

def gather_points(objs, depsgraph, samples=200000, seed=3):
    """Area-weighted random surface points of all objects (world space)."""
    tris_all, areas = [], []
    for o in objs:
        A = C.mesh_arrays(o, depsgraph)
        co = A.world_co()
        if len(A.tris) == 0:
            continue
        t = co[A.tris]                                   # (n, 3, 3)
        a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        tris_all.append(t)
        areas.append(a)
    if not tris_all:
        return np.zeros((0, 3))
    T = np.concatenate(tris_all)
    A = np.concatenate(areas)
    if A.sum() <= 0:
        return T.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(T), size=samples, p=A / A.sum())
    r1, r2 = rng.random(samples), rng.random(samples)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    t = T[pick]
    return t[:, 0] + (t[:, 1] - t[:, 0]) * r1[:, None] + (t[:, 2] - t[:, 0]) * r2[:, None]


def card_frame(points, side="FRONT"):
    """Vertical frame: u along the main horizontal direction, z up, n = facing."""
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    cov = xy.T @ xy
    w, v = np.linalg.eigh(cov)
    d = v[:, np.argmax(w)]
    u = Vector((d[0], d[1], 0.0)).normalized()
    if abs(u.x) >= abs(u.y):
        if u.x < 0:
            u = -u
    elif u.y < 0:
        u = -u
    z = Vector((0, 0, 1))
    n = u.cross(z).normalized()           # quads (u0,z0)→(u1,z0)→(u1,z1) face +n
    if side == "BACK":
        u, n = -u, -n
    return u, z, n


def project(points, u, z, n):
    U = np.array(u)
    Z = np.array(z)
    N = np.array(n)
    return points @ U, points @ Z, points @ N


def find_period(pu, pz, length, min_lag=0.05):
    """Dominant repeat step along u. Returns (period or None, score 0..1)."""
    if length <= 0 or len(pu) < 100:
        return None, 0.0
    nb = int(min(8192, max(256, length / 0.005)))
    bins = nb
    zb = 24
    h, _, _ = np.histogram2d(pu, pz, bins=(bins, zb))
    h = h - h.mean(axis=0, keepdims=True)
    # autocorrelation along u (zero padded → linear, not circular)
    F = np.fft.rfft(h, n=2 * bins, axis=0)
    ac = np.fft.irfft(F * np.conj(F), axis=0)[:bins].sum(axis=1)
    overlap = (bins - np.arange(bins)) / bins          # unbiased estimate
    ac = ac / np.maximum(overlap, 1e-6)
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    step = length / bins
    lo = max(2, int(min_lag / step))
    hi = int(bins / 2)                                # need ≥ 2 repeats
    if hi <= lo + 2:
        return None, 0.0
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(seg[i] for i in peaks)
    if best < 0.4:
        return None, float(best)
    cands = period_candidates(ac, lo, seg, peaks, best, step)
    return cands[0]


def _refine(ac, k):
    if 1 <= k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if den != 0 else 0.0
    return 0.0


def period_candidates(ac, lo, seg, peaks, best, step):
    """Peaks sorted by lag: the strongest-first rule picks long periods when a few odd
    parts (corner posts) break the short one, so every reasonable peak is offered."""
    out = []
    first = next(i for i in peaks if seg[i] >= 0.9 * best)
    for i in [first] + [i for i in peaks if i != first and seg[i] >= 0.3]:
        k = lo + i
        out.append(((k + _refine(ac, k)) * step, float(seg[i])))
    # shortest first, but keep the classic pick first among equals
    return sorted(out, key=lambda c: c[0])


def find_periods(pu, pz, length, min_lag=0.05):
    """All period candidates (shortest first)."""
    if length <= 0 or len(pu) < 100:
        return []
    nb = int(min(8192, max(256, length / 0.005)))
    h, _, _ = np.histogram2d(pu, pz, bins=(nb, 24))
    h = h - h.mean(axis=0, keepdims=True)
    F = np.fft.rfft(h, n=2 * nb, axis=0)
    ac = np.fft.irfft(F * np.conj(F), axis=0)[:nb].sum(axis=1)
    ac = ac / np.maximum((nb - np.arange(nb)) / nb, 1e-6)
    if ac[0] <= 0:
        return []
    ac = ac / ac[0]
    step = length / nb
    lo = max(2, int(min_lag / step))
    hi = int(nb / 2)
    if hi <= lo + 2:
        return []
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return []
    best = max(seg[i] for i in peaks)
    if best < 0.4:
        return []
    return period_candidates(ac, lo, seg, peaks, best, step)


def _douglas_peucker(x, y, tol):
    keep = np.zeros(len(x), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(x) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        xa, ya, xb, yb = x[a], y[a], x[b], y[b]
        t = (x[a + 1:b] - xa) / (xb - xa) if xb != xa else np.zeros(b - a - 1)
        dev = np.abs(y[a + 1:b] - (ya + t * (yb - ya)))
        i = int(np.argmax(dev))
        if dev[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return x[keep], y[keep]


def _bin_percentile(pu, pz, u0, u1, cell, q):
    nb = max(4, int(math.ceil((u1 - u0) / cell)))
    idx = np.clip(((pu - u0) / cell).astype(int), 0, nb - 1)
    order = np.argsort(idx, kind="stable")
    idx_s, z_s = idx[order], pz[order]
    starts = np.searchsorted(idx_s, np.arange(nb))
    ends = np.searchsorted(idx_s, np.arange(nb), side="right")
    out = np.full(nb, np.nan)
    for b_ in range(nb):
        if ends[b_] > starts[b_]:
            out[b_] = np.percentile(z_s[starts[b_]:ends[b_]], q)
    ok = ~np.isnan(out)
    xs = u0 + (np.arange(nb) + 0.5) * cell
    if ok.sum() >= 2:
        out = np.interp(xs, xs[ok], out[ok])
    else:
        out = np.zeros(nb)
    return xs, out


def _filt(a, k, fn):
    k = max(1, int(k)) | 1
    # mirror at the ends: a post standing right at the end stays a narrow spike
    pad = np.pad(a, k // 2, mode="reflect" if len(a) > k // 2 else "edge")
    return fn(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)


def _closing(a, k):   # fills dips narrower than k
    return _filt(_filt(a, k, np.max), k, np.min)


def _opening(a, k):   # removes peaks narrower than k
    return _filt(_filt(a, k, np.min), k, np.max)


def _ends(a, k):
    """Posts standing right at the ends look like wide steps to the filters; continue
    the line from just inside instead."""
    n = len(a)
    if n < 4 * k + 2:
        return a
    a = a.copy()
    i0, i1 = k, 2 * k
    slope = (a[i1] - a[i0]) / (i1 - i0)
    a[:k] = a[i0] - slope * (i0 - np.arange(k))
    j0, j1 = n - 1 - k, n - 1 - 2 * k
    slope = (a[j0] - a[j1]) / (j0 - j1)
    a[n - k:] = a[j0] + slope * (np.arange(n - k, n) - j0)
    return a


def top_profile(pu, pz, u0, u1, cell=0.05, tol=0.03):
    """Profile of a railing that follows stairs: breakpoints (bu, bz) of its top edge,
    or None when the object is not a constant-height band going up and down
    (a flat fence, a jagged facade…)."""
    xs, top = _bin_percentile(pu, pz, u0, u1, cell, 98)
    _, bot = _bin_percentile(pu, pz, u0, u1, cell, 2)
    gap = int(round(0.5 / cell))       # gaps between pickets / balusters
    spike = int(round(0.3 / cell))     # posts, single bars
    top = _ends(_opening(_closing(top, gap), spike), spike)
    bot = _ends(_closing(_opening(bot, gap), spike), spike)
    band = top - bot
    # stretches where only the handrail runs (over a landing, without bars below) have
    # no band at all: judge the band where it exists, the profile comes from the top
    full = band > 0.4 * float(np.percentile(band, 90))
    if full.mean() < 0.4:
        return None
    bh = float(np.median(band[full]))
    need = max(0.1, 0.25 * bh)
    if bh <= 0 or float(np.ptp(top)) < need or float(np.ptp(bot[full])) < need:
        return None                            # practically flat
    # most of the length must keep the band height (end gates / single posts may not)
    if float(np.percentile(np.abs(band[full] - bh), 75)) > 0.1 * bh:
        return None                            # the band doesn't keep its height
    dt, db = top[full] - top[full].mean(), bot[full] - bot[full].mean()
    if dt.std() == 0 or db.std() == 0 or float(np.corrcoef(dt, db)[0, 1]) < 0.85:
        return None                            # top and bottom don't rise together
    xs = np.concatenate([[u0], xs, [u1]])
    top = np.concatenate([[top[0]], top, [top[-1]]])
    prof = _douglas_peucker(xs, top, tol)
    # a stair goes only up (or only down) along one railing; a saw-tooth top is a gate,
    # a step in the fence or posts of different height — not a profile to follow
    dz = np.diff(np.asarray(prof[1], float))
    if (dz > 0.1).any() and (dz < -0.1).any():
        return None
    return prof


def validate_period(pu, pz, length, height, period, cell=0.1):
    """How periodic the object really is at *period* (0..1).

    The object is cut into windows of a whole number of periods (≥ 0.5 m); each
    window's surface-density map is compared with the median window. The score is the
    median similarity, so a few odd places (corner posts, a brace) don't spoil a
    periodic railing, while brick-like / random layouts score low."""
    if not period or length < 2 * period:
        return 0.0
    kp = max(1, math.ceil(0.5 / period - 1e-9))
    W = kp * period
    nwin = int(length // W)
    cell = min(cell, period / 6)
    nz = int(min(10, max(4, height / 0.1)))
    if nwin < 3:
        nu = int(min(20000, max(20, length / cell)))
        h, _, _ = np.histogram2d(pu, pz, bins=(nu, nz))
        k = int(round(period / (length / nu)))
        if k <= 0 or k >= nu:
            return 0.0
        a_, b_ = h[:-k].ravel(), h[k:].ravel()
        if a_.std() == 0 or b_.std() == 0:
            return 0.0
        return float(np.corrcoef(a_, b_)[0, 1])
    nu = max(6, int(round(W / cell)))
    zr = (float(pz.min()), float(pz.max()) + 1e-9)
    wins = []
    for j in range(nwin):
        a0 = j * W
        m = (pu >= a0) & (pu < a0 + W)
        h, _, _ = np.histogram2d(pu[m] - a0, pz[m], bins=(nu, nz), range=((0, W), zr))
        wins.append(h.ravel())
    Wn = np.array(wins)
    med = np.median(Wn, axis=0)
    if med.std() == 0:
        return 0.0
    scores = []
    for w in Wn:
        scores.append(0.0 if w.std() == 0 else float(np.corrcoef(w, med)[0, 1]))
    return float(np.median(scores))


def area_rect(tile, model_type, quadrant):
    if model_type == "VPM":
        return (0, 0, tile, tile)
    cx, cy = QUADRANTS[quadrant]
    half = tile // 2
    return (cx * half, cy * half, half, half)


GUTTER = 4  # px between packed strips, so filtering never mixes neighbours


def pack_layout(length, height, d, aw, ah):
    """Cut a non-repeating object into strips that fit the area at density *d*.
    Returns (pieces, tiles) or None if a single strip can't fit. A piece is
    (u_a, u_b, z_a, z_b, tile_index, px_x, px_y, w_px, h_px) relative to the object."""
    band_px = min(ah, math.ceil(height * d))
    if band_px < 1:
        return None
    band_m = band_px / d
    chunk_m = aw / d
    nz = max(1, math.ceil(height / band_m - 1e-9))
    nu = max(1, math.ceil(length / chunk_m - 1e-9))
    shelf = band_px + GUTTER
    per_tile = max(1, (ah + GUTTER) // shelf)
    pieces = []
    k = 0
    for iz in range(nz):
        za = iz * band_m
        zb = min(height, za + band_m)
        for iu in range(nu):
            ua = iu * chunk_m
            ub = min(length, ua + chunk_m)
            t, sh = divmod(k, per_tile)
            w = max(1, min(aw, round((ub - ua) * d)))
            h = max(1, min(band_px, round((zb - za) * d)))
            pieces.append((ua, ub, za, zb, t, 0, sh * shelf, w, h))
            k += 1
    tiles = pieces[-1][4] + 1
    return pieces, tiles


def pack_runs(items, d, aw, ah):
    """Strips of several objects in ONE area at density *d* (shelf packing, tallest first).
    *items*: [(key, length, height)]. Returns {key: [piece, …]} with pieces as in
    pack_layout (tile index 0) or None if they don't fit."""
    strips = []
    for key, length, height in items:
        band_px = min(ah, math.ceil(height * d))
        if band_px < 1:
            return None
        band_m = band_px / d
        chunk_m = (aw - GUTTER) / d
        nz = max(1, math.ceil(height / band_m - 1e-9))
        nu = max(1, math.ceil(length / chunk_m - 1e-9))
        for iz in range(nz):
            za, zb = iz * band_m, min(height, (iz + 1) * band_m)
            for iu in range(nu):
                ua, ub = iu * chunk_m, min(length, (iu + 1) * chunk_m)
                w = max(1, math.ceil((ub - ua) * d))
                h = max(1, math.ceil((zb - za) * d))
                strips.append((key, ua, ub, za, zb, w, h))
    strips.sort(key=lambda t: (-t[6], -t[5]))
    out = {}
    x = y = 0
    shelf_h = 0
    g = GUTTER
    for key, ua, ub, za, zb, w, h in strips:
        if x + w > aw:
            x, y = 0, y + shelf_h + g
            shelf_h = 0
        if y + h > ah or w > aw:
            return None
        out.setdefault(key, []).append((ua, ub, za, zb, 0, x, y, w, h))
        x += w + g
        shelf_h = max(shelf_h, h)
    return out


def best_run_packing(items, aw, ah, dmin, dmax):
    """Highest density in [dmin, dmax] at which all *items* fit one area."""
    d = dmax
    while d >= dmin - 1e-9:
        r = pack_runs(items, d, aw, ah)
        if r is not None:
            return d, r
        d *= 0.97
    r = pack_runs(items, dmin, aw, ah)
    return (dmin, r) if r is not None else (None, None)


def piece_geometry(info, pieces, area, tile, d):
    """Quads of one object's *pieces* on its bake plane. Returns (verts, faces, uv_final,
    uv_area): final UVs in the atlas (half a pixel inside each rectangle) and UVs local to
    the area, used for the (supersampled) bake."""
    u, z, n = info["u"], info["z"], info["n"]
    x0, y0, aw, ah = area
    eps = max(1e-4, info["thickness"] * 0.02)
    nc = info["n1"] + eps
    verts, faces, uvf, uva = [], [], [], []
    hp = 0.5 / tile
    for (ua, ub, za, zb, _t, px, py, w, h) in pieces:
        a_, b_ = info["u0"] + ua, info["u0"] + ub
        c0, c1 = info["z0"] + za, info["z0"] + zb
        j = len(verts)
        for (uu, zz) in ((a_, c0), (b_, c0), (b_, c1), (a_, c1)):
            q = frame_point(u, z, n, uu, zz, nc)
            verts.append((q.x, q.y, q.z))
        faces.append((j, j + 1, j + 2, j + 3))
        wx, hy = (ub - ua) * d, (zb - za) * d               # exact, not whole pixels
        U0, V0 = (x0 + px) / tile, (y0 + py) / tile
        U1, V1 = (x0 + px + wx) / tile, (y0 + py + hy) / tile
        uvf += [(U0 + hp, V0 + hp), (U1 - hp, V0 + hp), (U1 - hp, V1 - hp), (U0 + hp, V1 - hp)]
        b0, c0_, b1, c1_ = px / aw, py / ah, (px + wx) / aw, (py + hy) / ah
        uva += [(b0, c0_), (b1, c0_), (b1, c1_), (b0, c1_)]
    return verts, faces, uvf, uva


def joined_copy(context, items, name="TK_npm_sources", offsets=None):
    """One object holding world-space copies of all *items* [(obj, info|None)] — when info
    has a profile the copy is straightened. Cycles' selected-to-active casts rays against
    every selected object separately, so one joined object bakes many times faster."""
    dg = context.evaluated_depsgraph_get()
    cos, lvs, lss, lts, mis, uvs, smooth = [], [], [], [], [], [], []
    mats = []
    vo = lo = 0
    for o, inf in items:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), depsgraph=dg)
        try:
            me.transform(o.matrix_world)
            nv, nl, npo = len(me.vertices), len(me.loops), len(me.polygons)
            co = np.empty(nv * 3)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            if inf is not None and inf.get("profile") is not None:
                bu, bz = inf["profile"]
                co -= np.interp(co @ np.array(inf["u"]), bu, bz)[:, None] * np.array(inf["z"])[None, :]
            if offsets is not None:
                co += np.asarray(offsets[len(cos)])[None, :]
            lv = np.empty(nl, np.int64); me.loops.foreach_get("vertex_index", lv)
            ls = np.empty(npo, np.int64); me.polygons.foreach_get("loop_start", ls)
            lt = np.empty(npo, np.int64); me.polygons.foreach_get("loop_total", lt)
            mi = np.empty(npo, np.int64); me.polygons.foreach_get("material_index", mi)
            sm = np.empty(npo, bool); me.polygons.foreach_get("use_smooth", sm)
            remap = []
            for m in (me.materials or [None]):
                if m not in mats:
                    mats.append(m)
                remap.append(mats.index(m))
            mi = np.array(remap)[np.clip(mi, 0, len(remap) - 1)]
            uv = np.zeros((nl, 2), np.float32)
            if me.uv_layers:
                me.uv_layers[0].data.foreach_get("uv", uv.ravel())
            cos.append(co); lvs.append(lv + vo); lss.append(ls + lo); lts.append(lt)
            mis.append(mi); uvs.append(uv); smooth.append(sm)
            vo += nv
            lo += nl
        finally:
            bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name)
    co = np.concatenate(cos); lv = np.concatenate(lvs); ls = np.concatenate(lss)
    lt = np.concatenate(lts)
    me.vertices.add(len(co)); me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.loops.add(len(lv)); me.loops.foreach_set("vertex_index", lv.astype(np.int32))
    me.polygons.add(len(ls)); me.polygons.foreach_set("loop_start", ls.astype(np.int32))
    try:
        me.polygons.foreach_set("loop_total", lt.astype(np.int32))
    except (AttributeError, RuntimeError, TypeError):
        pass
    me.polygons.foreach_set("material_index", np.concatenate(mis).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.concatenate(smooth))
    for m in mats:
        me.materials.append(m)
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.concatenate(uvs).ravel())
    me.update(calc_edges=True)
    ob = bpy.data.objects.new(name, me)
    context.scene.collection.objects.link(ob)
    return ob


def thick_of(chunk):
    return max(pr[4][1]["thickness"] for pr in chunk)


def mesh_from(name, verts, faces, uvs):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", np.array(uvs, np.float32).ravel())
    return me


def bake_area(context, s, sources, target, area, d, thickness):
    """Bake all channels of *sources* onto *target* whose UVs cover the area in [0, 1].
    Supersampled at low density (a 2 cm bar at 10–40 px/m is under a pixel: averaging
    several samples gives it partial alpha instead of hit-or-miss). Returns {ch: array}."""
    _x0, _y0, aw, ah = area
    ss = 1 if d >= 150 else (2 if d >= 25 else 3)
    while ss > 1 and aw * ss > 3072:                  # one float image ≤ ~150 MB
        ss -= 1
    eps = max(1e-4, thickness * 0.02)
    margin = max(4, int(s.bleed_px)) * ss

    def down(a):
        return a.reshape(ah, ss, aw, ss, a.shape[-1]).mean(axis=(1, 3))
    chans = list(s.channels(sources))
    chans.sort(key=lambda c: c != "OPACITY")       # alpha first: it weights the colours
    out, w_ = {}, None
    for ch in chans:
        yield ch
        a = bake_targets(context, sources, target, ch, aw * ss, eps, thickness + 3 * eps,
                         margin, s.ao_samples)[0]
        if ss == 1:
            out[ch] = a
        elif ch == "OPACITY":
            w_ = a[..., :1].copy()
            out[ch] = down(a)
        elif w_ is None:
            out[ch] = down(a)
        else:                                       # colour of what is there, not the void
            a *= w_
            num = down(a)
            den = down(w_)
            out[ch] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), num)
        a = None
    return out


def best_packing(length, height, aw, ah, dmin, dmax, max_tiles):
    """Highest density ≤ dmax whose strip layout fits in *max_tiles* areas."""
    d = dmax
    while d > 0.5:
        r = pack_layout(length, height, d, aw, ah)
        if r is not None and r[1] <= max_tiles:
            return d, r[0], r[1]
        d *= 0.985
    r = pack_layout(length, height, 0.5, aw, ah)
    return 0.5, r[0], r[1]


def tiles_needed(length, height, aw, ah, d):
    r = pack_layout(length, height, d, aw, ah)
    return r[1] if r else None


def plan(length, height, period, tile, model_type, dmin, dmax, quadrant="LT", max_tiles=1):
    """Work out texture area, density, segment length and repeats.

    Returns dict: area (x0, y0, w, h) in px, density, seg_len, spans, repeats, tiled,
    wrap (whole tile → UV wrap), used_h_px, warnings.
    """
    warns = []
    area = area_rect(tile, model_type, quadrant)
    aw, ah = area[2], area[3]
    dcap = dmax
    height_warn = None
    if height > 0:
        dcap = min(dcap, ah / height)
        if dcap < dmin:
            height_warn = (f"высота {height:.2f} м не помещается в {ah} px при "
                           f"{dmin:g} px/м — плотность снижена до {dcap:.1f} px/м")
    tiled = False
    spans = 1
    if period and length / period >= 2.0:
        spans = max(1, math.ceil(aw / (period * dcap) - 1e-9))
        seg = spans * period
        if seg < length - 1e-6:
            tiled = True
            d = aw / seg
            if d < dmin - 1e-6:
                warns.append(f"один пролёт {period:.2f} м при {dmin:g} px/м не влезает в "
                             f"{aw} px — плотность {d:.1f} px/м")
        else:
            seg = length
    else:
        seg = length
    pieces, ntiles = None, 1
    if tiled and height_warn:
        warns.append(height_warn)
    if not tiled:
        mt = max_tiles if model_type == "VPM" else 1
        d, pieces, ntiles = best_packing(length, height, aw, ah, dmin, dmax, mt)
        if d < dmin - 1e-6:
            need = tiles_needed(length, height, aw, ah, dmin)
            warns.append(f"без повтора {length:.1f}×{height:.1f} м при ≥{dmin:g} px/м нужно "
                         f"{need} тайл(ов) {aw}×{ah}; сейчас {d:.1f} px/м"
                         + (" — увеличьте «Максимум тайлов»" if model_type == "VPM" else ""))
    used_w = aw if tiled else max(p[5] + p[7] for p in pieces)
    used_h = (max(1, min(ah, math.ceil(height * d))) if tiled
              else max(p[6] + p[8] for p in pieces))
    return {
        "area": area, "density": d, "seg_len": seg, "spans": spans if tiled else 0,
        "repeats": length / seg if seg > 0 else 1.0, "tiled": tiled,
        "wrap": tiled and model_type == "VPM", "used_w": used_w, "used_h": used_h,
        "pieces": pieces, "tiles": ntiles, "warnings": warns,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def frame_point(u, z, n, a, b, c):
    return u * a + z * b + n * c


def make_quad_object(name, corners, uvs, collection, material=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(c) for c in corners], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for i, d in enumerate(uv.data):
        d.uv = uvs[i]
    if material is not None:
        me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def build_card_mesh(name, u, z, n, u0, u1, z0, z1, nc, s0, seg, uv_rect, wrap, collection,
                    material):
    """Card over [u0,u1]×[z0,z1]. Texture phase: U = (u - s0) / seg (mod 1).

    uv_rect = (x0, y0, x1, y1) in 0..1 for the used part of the texture area.
    wrap=True → one quad with U beyond 0..1 (repeat); else a strip split at segment
    boundaries so every quad stays inside the area (atlas-safe).
    """
    x0, y0, x1, y1 = uv_rect
    verts, faces, uvs = [], [], []

    def V(a, b):
        p = frame_point(u, z, n, a, b, nc)
        return (p.x, p.y, p.z)

    if wrap:
        cuts = [u0, u1]
    else:
        k0 = math.floor((u0 - s0) / seg)
        cuts = [u0]
        k = k0 + 1
        while s0 + k * seg < u1 - 1e-6:
            cuts.append(s0 + k * seg)
            k += 1
        cuts.append(u1)
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if wrap:
            ua, ub = (a - s0) / seg, (b - s0) / seg
            U0 = x0 + ua * (x1 - x0)
            U1 = x0 + ub * (x1 - x0)
        else:
            base = math.floor((a - s0) / seg + 1e-9)
            fa = (a - s0) / seg - base
            fb = (b - s0) / seg - base
            U0 = x0 + fa * (x1 - x0)
            U1 = x0 + fb * (x1 - x0)
        j = len(verts)
        verts += [V(a, z0), V(b, z0), V(b, z1), V(a, z1)]
        faces.append((j, j + 1, j + 2, j + 3))
        uvs += [(U0, y0), (U1, y0), (U1, y1), (U0, y1)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uvl = me.uv_layers.new(name="UVMap")
    for i, d in enumerate(uvl.data):
        d.uv = uvs[i]
    if not wrap:
        # weld the strip so it is one continuous surface (UVs stay split)
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(me)
        bm.free()
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


# ---------------------------------------------------------------------------
# Atlas images & material
# ---------------------------------------------------------------------------

def atlas_image(atlas, ch, tile):
    name = f"{atlas}{M.CHANNELS[ch][1]}"
    img = bpy.data.images.get(name)
    if img is not None and tuple(img.size) == (tile, tile):
        return img
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, tile, tile, alpha=(ch == "DIFFUSE"))
    fill = (0, 0, 0) if ch == "OPACITY" else M.DEFAULT_FILL[ch]
    img.generated_color = (*fill, 1.0)
    if not M.CHANNELS[ch][3]:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    return img


def card_material(atlas, images):
    name = f"{atlas}_card"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        try:
            mat.use_nodes = True
        except Exception:
            pass
        M.wire_images(mat, images)
        mat.use_backface_culling = False
        M.set_transparency_mode(mat)
    else:
        # images are reused by name, so an existing material already points at them;
        # make sure channels added later get connected too
        have = {n.image.name for n in mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeTexImage" and n.image}
        missing = {ch: im for ch, im in images.items() if im.name not in have}
        if missing:
            M.wire_images(mat, missing, x=-900)
    return mat


# ---------------------------------------------------------------------------
# Bake
# ---------------------------------------------------------------------------

def _bake_image(ch, tile, name):
    img = bpy.data.images.new(name, tile, tile, alpha=True)
    if ch == "OPACITY":
        img.generated_color = (0, 0, 0, 1)          # misses stay 0 = transparent
    else:
        f = M.DEFAULT_FILL[ch]
        img.generated_color = (*f, 1.0)
    if not M.CHANNELS[ch][3]:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    return img


def bake_targets(context, sources, target, ch, tile, extrusion, max_dist, margin, ao_samples,
                 aa=1):
    """Bake *ch* from *sources* onto *target* (selected-to-active). Every material slot of
    the target gets its own tile×tile image. Returns a list of arrays, one per slot."""
    tmats = [sl.material for sl in target.material_slots]
    imgs, nodes = [], []
    for i, pm in enumerate(tmats):
        img = _bake_image(ch, tile, f"TK_card_{ch}_{i}")
        tex = pm.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = img
        pm.node_tree.nodes.active = tex
        imgs.append(img)
        nodes.append((pm, tex))
    kind = M.CHANNELS[ch][4]
    mats = []
    for o in sources:
        for slot in getattr(o, "material_slots", []):
            if slot.material is not None and slot.material not in mats:
                mats.append(slot.material)
    undos = [M.ensure_nodes(m) for m in mats]
    patches = []
    try:
        if kind == "EMIT":
            patches = [M._MatPatch(m, ch, None) for m in mats]
        with M._SceneForBake(context, target, int(ao_samples) if kind == "AO" else int(aa),
                             extra=sources):
            kw = dict(type=kind, use_selected_to_active=True, use_cage=False,
                      cage_extrusion=extrusion, max_ray_distance=max_dist,
                      margin=0 if ch == "OPACITY" else margin, use_clear=False,
                      target="IMAGE_TEXTURES")
            if kind == "NORMAL":
                kw["normal_space"] = "TANGENT"
            try:
                bpy.ops.object.bake(margin_type="EXTEND", **kw)
            except TypeError:
                bpy.ops.object.bake(**kw)
        arrays = [M.image_array(img).copy() for img in imgs]
    finally:
        for p in patches:
            p.restore()
        for u_ in undos:
            u_()
        for pm, tex in nodes:
            pm.node_tree.nodes.remove(tex)
        for img in imgs:
            bpy.data.images.remove(img)
    return arrays


def aa_samples(density):
    """Low densities (НПМ): a 2 cm bar is under a pixel — several samples per pixel give
    it partial alpha instead of a hit-or-miss line."""
    return 16 if density < 150 else 1


def bake_segment(context, sources, plane, ch, tile, extrusion, max_dist, margin, ao_samples,
                 aa=1):
    return bake_targets(context, sources, plane, ch, tile, extrusion, max_dist, margin,
                        ao_samples, aa)[0]


def finish_maps(crop, s):
    """Flora-style post-processing on one area (dict ch -> array). Returns opacity max."""
    op = crop["OPACITY"][..., 0]
    # НПМ: a thin bar is under a pixel and bakes half-transparent — a low threshold keeps
    # it as a solid pixel instead of losing it
    thr = min(s.threshold, 0.4) if s.model_type == "NPM" else s.threshold
    if s.bleed and "DIFFUSE" in crop:
        d = crop["DIFFUSE"]
        d[..., :3] = M.bleed_colors(d[..., :3], op >= thr, int(s.bleed_px))
    if s.binary_mask:
        op = (op >= thr).astype(np.float32)
    crop["OPACITY"][..., 0] = crop["OPACITY"][..., 1] = crop["OPACITY"][..., 2] = op
    if "DIFFUSE" in crop:
        crop["DIFFUSE"][..., 3] = op
    return float(op.max())


def write_atlas(atlas, crop, tile, sl):
    images = {}
    for ch, c in crop.items():
        img = atlas_image(atlas, ch, tile)
        full = M.image_array(img).copy()
        full[sl] = c
        if ch != "DIFFUSE":
            full[sl][..., 3] = 1.0
        M.set_image_array(img, full)
        try:
            img.pack()
        except RuntimeError:
            pass
        images[ch] = img
    return images


def fit_local_period(pu, a, b, P0, rel=0.12, nfold=64):
    """Best period near *P0* and the position of a bar (densest column) inside [a, b).
    Returns (P, phase_u, sharpness) or (P0, None, 0) when the section is too short."""
    m = (pu >= a) & (pu < b)
    x = pu[m]
    if len(x) < 300 or (b - a) < 1.5 * P0:
        return P0, None, 0.0
    cands = np.linspace(P0 * (1 - rel), P0 * (1 + rel), 241)
    best = (0.0, P0, 0)
    for P in cands:
        h = np.bincount((((x - a) / P) % 1.0 * nfold).astype(int), minlength=nfold)
        sc = h.max() / max(h.mean(), 1e-9)
        if sc > best[0]:
            best = (sc, P, int(np.argmax(h)))
    sc, P, k = best
    return float(P), float(a + (k + 0.5) / nfold * P), float(sc)


def _sharpness_many(x, a, cands, nfold=64, block=48):
    """_sharpness for many periods at once (vectorised in blocks)."""
    sc = np.empty(len(cands))
    ks = np.empty(len(cands), int)
    xa = (x - a).astype(np.float64)
    for i0 in range(0, len(cands), block):
        P = np.asarray(cands[i0:i0 + block], np.float64)
        f = np.mod(xa[None, :] / P[:, None], 1.0)
        idx = np.minimum((f * nfold).astype(np.int64), nfold - 1)
        idx += (np.arange(len(P)) * nfold)[:, None]
        h = np.bincount(idx.ravel(), minlength=len(P) * nfold).reshape(len(P), nfold)
        sc[i0:i0 + len(P)] = h.max(1) / np.maximum(h.mean(1), 1e-9)
        ks[i0:i0 + len(P)] = h.argmax(1)
    return sc, ks


def _sharpness(x, a, P, nfold=64):
    h = np.bincount((((x - a) / P) % 1.0 * nfold).astype(int), minlength=nfold)
    return h.max() / max(h.mean(), 1e-9), int(np.argmax(h))


def bars_per_period(pu, a, b, P):
    """How many equally spaced bars one period holds (1…4): folding at P/m keeps the
    bars stacked only while m divides their count."""
    m_ = (pu >= a) & (pu < b)
    x = pu[m_]
    if len(x) < 300:
        return 1
    nf = 64
    s1, _ = _sharpness(x, a, P, nf)
    h1 = np.bincount((((x - a) / P) % 1.0 * nf).astype(int), minlength=nf).astype(float)
    k1 = int(np.argmax(h1))
    def slot(k):
        return sum(h1[(k + d) % nf] for d in (-1, 0, 1))
    for m in (4, 3, 2):
        sm, _ = _sharpness(x, a, P / m)
        if sm < 0.8 * s1:
            continue
        # the m bars of one period must be alike (a post among pickets is not "m bars")
        v = [slot(int(round(k1 + i * nf / m))) for i in range(m)]
        if min(v) >= 0.75 * max(v):
            return m
    return 1


def fit_bar_step(pu, a, b, step0, lo=0.55, hi=1.8, nfold=64):
    """Bar step of a section, searched widely around *step0*. Sub-multiples of the true
    step stack the bars just as well, so the LARGEST step that still stacks them wins.
    Returns (step, phase_u, sharpness) or (step0, None, 0)."""
    m_ = (pu >= a) & (pu < b)
    x = pu[m_]
    if len(x) < 300 or (b - a) < 2.5 * step0:
        return step0, None, 0.0
    if len(x) > 40000:                       # plenty for a histogram; keeps the UI alive
        x = x[np.random.default_rng(0).choice(len(x), 40000, replace=False)]
    cands = np.linspace(step0 * lo, step0 * hi, 500)
    sc, ks = _sharpness_many(x, a, cands, nfold)
    good = np.nonzero(sc >= 0.8 * sc.max())[0]
    # among the good ones take the largest step, refined to its local maximum
    j = good[-1]
    while j > 0 and sc[j - 1] > sc[j]:
        j -= 1
    lo_j = j
    while lo_j > 0 and sc[lo_j - 1] >= 0.8 * sc.max():
        lo_j -= 1
    j = lo_j + int(np.argmax(sc[lo_j:good[-1] + 1]))
    P = float(cands[j])
    return P, float(a + (ks[j] + 0.5) / nfold * P), float(sc[j])


def fit_sections(info, P_ref, m=None):
    """Per-section (between profile breakpoints) period (= m bar steps) and bar phase."""
    pu, _pz = info["samples"]
    if m is None:
        m = info.get("bars_per_period", 1)
    step0 = P_ref / m
    if info.get("profile") is not None:
        cuts = list(info["profile"][0])
    else:
        cuts = [info["u0"], info["u1"]]
    secs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        st, ph, sc = fit_bar_step(pu, a, b, step0)
        secs.append({"a": float(a), "b": float(b), "P": st * m, "phase": ph, "score": sc})
    # sections too short to measure take the phase from a neighbour and the reference step
    for i, sct in enumerate(secs):
        if sct["phase"] is None:
            sct["P"] = P_ref
            nb = next((secs[j] for j in (i - 1, i + 1) if 0 <= j < len(secs)
                       and secs[j]["phase"] is not None), None)
            sct["phase"] = nb["phase"] if nb else info["u0"]
    return secs


def typical_segment(info, seg):
    """Start of the most typical seg-long window (aligned to the period), so odd parts
    like corner posts or a brace don't get repeated along the whole card."""
    per = info["period"]
    pu, pz = info["samples"]
    u0, length = info["u0"], info["length"]
    n = int((length - seg) / per) + 1
    if n <= 1:
        return u0
    cell = min(per / 8, 0.02)
    nu = max(8, int(round(seg / cell)))
    nz = 24
    wins = []
    for j in range(n):
        a = u0 + j * per
        m = (pu >= a) & (pu < a + seg)
        h, _, _ = np.histogram2d(pu[m] - a, pz[m], bins=(nu, nz),
                                 range=((0, seg), (info["z0"], info["z1"])))
        wins.append(h)
    W = np.array(wins)
    mass = W.sum(axis=(1, 2))
    ok = mass >= 0.7 * np.median(mass)            # skip sparse windows (ends, gaps)
    if n > 4:
        ok[0] = ok[-1] = False                    # and the very ends (returns, end posts)
    if not ok.any():
        ok[:] = True
    med = np.median(W[ok], axis=0)
    dev = np.abs(W - med).sum(axis=(1, 2)) / np.maximum(mass, 1)
    dev[~ok] = np.inf
    j = int(np.argmin(dev))
    return u0 + j * per


def straightened_copies(context, objs, info):
    """Evaluated copies of *objs* with the profile removed: z -= profile(u)."""
    bu, bz = info["profile"]
    U = np.array(info["u"])
    Zv = np.array(info["z"])
    dg = context.evaluated_depsgraph_get()
    out = []
    for o in objs:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), depsgraph=dg)
        me.transform(o.matrix_world)
        n = len(me.vertices)
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        co -= np.interp(co @ U, bu, bz)[:, None] * Zv[None, :]
        me.vertices.foreach_set("co", co.ravel())
        me.update()
        t = bpy.data.objects.new(f"TK_straight_{o.name}", me)
        context.scene.collection.objects.link(t)
        out.append(t)
    return out


def detect_bars(info, step0):
    """u positions of the vertical bars (balusters, posts), from the surface density in the
    middle of the band, between the rails. Straightened space."""
    pu, pz = info["samples"]
    z0, z1 = info["z0"], info["z1"]
    H = z1 - z0
    m = (pz > z0 + 0.25 * H) & (pz < z1 - 0.25 * H)
    x = pu[m]
    if len(x) < 200:
        return np.zeros(0)
    cell = min(0.005, step0 / 20)
    nb = max(10, int(math.ceil((info["u1"] - info["u0"]) / cell)))
    h, e = np.histogram(x, bins=nb, range=(info["u0"], info["u1"]))
    w = max(3, int(round(min(0.25 * step0, 0.03) / cell)) | 1)   # merges the sides of a bar
    h = np.convolve(h.astype(float), np.ones(w) / w, mode="same")
    thr = 0.25 * np.percentile(h[h > 0], 95) if (h > 0).any() else 1
    pk = np.nonzero((h[1:-1] >= h[:-2]) & (h[1:-1] > h[2:]) & (h[1:-1] > thr))[0] + 1
    if len(pk) == 0:
        return np.zeros(0)
    # one peak per bar: strongest first, nothing else within 60 % of a step
    rad = 0.6 * step0 / cell
    # posts on both sides of a gap are separate bars: suppress only within one piece
    gl = find_gaps(info)
    cuts = [g[0] for g in gl]
    def piece(q):
        return int(np.searchsorted(cuts, e[q]))
    taken = []
    for q in pk[np.argsort(-h[pk], kind="stable")]:
        if all(abs(q - t) >= rad or piece(q) != piece(t) for t in taken):
            taken.append(q)
    taken.sort()
    c = (e[:-1] + e[1:]) / 2
    # bar centre = density centroid around the peak (thick posts have two side faces)
    r = max(1, int(min(0.2 * step0, 0.15) / cell))
    raw = np.histogram(x, bins=nb, range=(info["u0"], info["u1"]))[0].astype(float)
    out = []
    for q in taken:
        k_ = piece(q)                       # the window must not reach across a gap
        plo = 0 if k_ == 0 else int(np.searchsorted(e, gl[k_ - 1][1]))
        phi = nb if k_ >= len(gl) else int(np.searchsorted(e, gl[k_][0]))
        for _ in range(2):
            lo, hi = max(plo, q - r), min(phi, q + r + 1)
            wsum = raw[lo:hi].sum()
            if wsum <= 0:
                break
            q = int(round((raw[lo:hi] * np.arange(lo, hi)).sum() / wsum))
        out.append(float(c[min(max(q, 0), nb - 1)]))
    return np.array(out)


def bar_grid(peaks, step0, keep_last=False):
    """Turn detected peaks into a bar sequence with integer indices: off-grid peaks (posts at
    breaks, brackets) are dropped, missing bars are bridged. The local step may drift slowly
    (sections of one railing differ by a few per cent)."""
    if len(peaks) < 2:
        return np.zeros(0), np.zeros(0, int)
    xs, ix = [float(peaks[0])], [0]
    st = step0
    peaks = list(peaks)
    for x in peaks[1:]:
        g = x - xs[-1]
        n = g / st
        if n < 0.65:
            # two candidates for one slot: keep the one on the grid of the previous bar
            if len(xs) >= 2:
                ref = xs[-2] + st * (ix[-1] - ix[-2])
                if abs(x - ref) < abs(xs[-1] - ref):
                    xs[-1] = float(x)
            continue
        kk = int(round(n))
        if abs(n - kk) > 0.3 and not (keep_last and x is peaks[-1]):
            continue                                   # off-grid (a post at a gap still counts)
        kk = max(1, kk)
        xs.append(float(x))
        ix.append(ix[-1] + kk)
        if kk == 1:
            st = float(np.clip(0.7 * st + 0.3 * g, 0.8 * step0, 1.25 * step0))
    return np.array(xs), np.array(ix)


def bar_card_uvs(card, info, m, k, off=0.5):
    """Cut the card at every bar and map texture bars onto the real bars one by one:
    model bar with index j ↔ texture bar at t = off + (j + c)/m periods (U = t / k)."""
    import bmesh
    peaks = info.get("bars")
    if peaks is None or len(peaks) < 3:
        return False
    P = info["period"]
    gap = float(np.median(np.diff(peaks)))
    if not 0.8 * P / m < gap < 1.25 * P / m:
        return False                                   # peaks are not the bars of this step
    gaps = find_gaps(info)
    if gaps:
        # grid per piece; the index jumps over the gap
        cuts = [g[0] for g in gaps]
        pid = np.searchsorted(cuts, peaks)
        bl, il = [], []
        for q in np.unique(pid):
            b_, i_ = bar_grid(peaks[pid == q], P / m, keep_last=True)
            if len(b_) == 0:
                b_, i_ = peaks[pid == q][:1], np.zeros(1, int)
            if bl:
                jump = max(1, int(round((b_[0] - bl[-1][-1]) / (P / m))))
                i_ = i_ + il[-1][-1] + jump
            bl.append(b_)
            il.append(i_)
        bars, idx = np.concatenate(bl), np.concatenate(il)
    else:
        bars, idx = bar_grid(peaks, P / m)
    if len(bars) < 3 or len(bars) < 0.7 * len(peaks):
        return False
    # a perfectly regular railing needs no cuts: the card's own UVs already fit
    A = np.vstack([idx, np.ones(len(idx))]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, bars, rcond=None)
    resid = np.abs(bars - (slope * idx + icpt)).max()
    if abs(slope - P / m) < 0.003 * P / m and resid < max(0.004, 0.03 * P / m) \
            and info.get("profile") is None and (info.get("texture") or {}).get("s0") is not None:
        return True
    L = m * k
    tex = info.get("texture") or {}
    tb = None
    c = 0
    s0 = tex.get("s0")
    if s0 is not None and "tb" not in tex:
        # this run was baked: measure where the bars landed in the texture with the same
        # detector, so its bias cancels out
        inw = np.nonzero((bars >= s0 - 1e-6) & (bars < s0 + k * P - 1e-6))[0]
        if len(inw) == L and np.all(np.diff(idx[inw]) == 1):
            tb = (bars[inw] - s0) / P                 # in periods, 0…k
            c = -int(idx[inw[0]])
            info["tb"] = tb
    elif tex.get("tb") is not None and len(tex["tb"]) == L:
        tb = np.asarray(tex["tb"])
    if tb is None or m > 1:
        if m > 1:
            secs = fit_sections(info, P, m)
            sec = next((x for x in secs if x["a"] <= bars[0] <= x["b"]), secs[0])
            c0 = int(round((bars[0] - sec["phase"]) / sec["P"] * m)) % m
            if tb is None or s0 is None or "tb" in tex:
                c = c0
    if tb is None:
        tb = off + np.arange(L) / m

    def tex_t(g):
        """texture position (periods) of bar grid index g (real-valued)."""
        g0 = math.floor(g)
        f = g - g0
        def at(i):
            return tb[i % L] + (i // L) * k
        return at(g0) + f * (at(g0 + 1) - at(g0))
    U = Vector(info["u"])
    me = card.data
    bm = bmesh.new()
    bm.from_mesh(me)
    gaps_ = find_gaps(info)
    for x in list(bars) + [g for ab in gaps_ for g in ab]:
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=U * float(x), plane_no=U)
    uvl = bm.loops.layers.uv.active
    jj = idx.astype(float) + c

    def local_step(j):
        """metres per bar step around bar j, from its neighbour on the same side."""
        if 0 < j and idx[j] != idx[j - 1] and not any(bars[j - 1] < a_ < bars[j] for a_, _ in gaps_):
            return (bars[j] - bars[j - 1]) / (idx[j] - idx[j - 1])
        if j + 1 < len(bars) and not any(bars[j] < a_ < bars[j + 1] for a_, _ in gaps_):
            return (bars[j + 1] - bars[j]) / (idx[j + 1] - idx[j])
        return P / m

    def t_at(u, j):
        """u inside bar interval j (−1 = before the first bar, len−1 = after the last)."""
        if j < 0:
            g = (bars[1] - bars[0]) / (idx[1] - idx[0])
            return tex_t(jj[0]) + (u - bars[0]) / g / m
        if j >= len(bars) - 1:
            g = (bars[-1] - bars[-2]) / (idx[-1] - idx[-2])
            return tex_t(jj[-1]) + (u - bars[-1]) / g / m
        gap = next(((a_, b_) for a_, b_ in gaps_ if bars[j] < a_ < bars[j + 1]), None)
        if gap is not None:
            # the interval spans a gap: each side continues its own bar at its own step
            # (interpolating across the gap would squeeze the end posts)
            if u <= 0.5 * (gap[0] + gap[1]):
                return tex_t(jj[j]) + (u - bars[j]) / local_step(j) / m
            return tex_t(jj[j + 1]) - (bars[j + 1] - u) / local_step(j + 1) / m
        f = (u - bars[j]) / (bars[j + 1] - bars[j])
        return tex_t(jj[j] + f * (jj[j + 1] - jj[j]))

    for f in bm.faces:
        uc = f.calc_center_median().dot(U)
        j = int(np.searchsorted(bars, uc, side="right") - 1)
        for lp in f.loops:
            lp[uvl].uv.x = t_at(lp.vert.co.dot(U), j) / k
    bm.to_mesh(me)
    bm.free()
    me.update()
    return True


def fit_card_uvs(card, info):
    """Re-map U on every section so the texture repeat lands on that section's own bars."""
    import bmesh
    p = info["plan"]
    k = max(1, p["spans"])
    # smallest number of bars per period whose step the detected bars actually follow
    P = info["period"]
    for m in (range(1, 17) if not os.environ.get("TK_NOBARS") else ()):
        if P / m < 0.04:
            break
        bars = detect_bars(info, P / m)
        if len(bars) < 3:
            continue
        info["bars"] = bars
        if bar_card_uvs(card, info, m, k, info.get("bar_offset", 0.5)):
            if info.get("texture") is not None and info.get("tb") is not None:
                info["texture"]["tb"] = info["tb"]
                info["texture"]["m"] = m
            return
    secs = fit_sections(info, info["period"])
    info["sections"] = secs
    U = Vector(info["u"])
    me = card.data
    bm = bmesh.new()
    bm.from_mesh(me)
    if info.get("profile") is None:
        # no breakpoints yet: cut the card where sections change (here: nowhere)
        pass
    uvl = bm.loops.layers.uv.active
    off = info.get("bar_offset", 0.5)
    for f in bm.faces:
        uc = f.calc_center_median().dot(U)
        sec = next((x for x in secs if x["a"] - 1e-6 <= uc <= x["b"] + 1e-6), secs[-1])
        for lp in f.loops:
            u = lp.vert.co.dot(U)
            t = (u - sec["phase"]) / sec["P"] + off       # in periods; bars at t = off
            lp[uvl].uv.x = t / k
    bm.to_mesh(me)
    bm.free()
    me.update()


def find_gaps(info, min_gap=0.08, cell=0.02):
    """u-intervals with no geometry at all (two railings end to end with a gap between)."""
    pu, pz = info["samples"]
    u0, u1 = info["u0"], info["u1"]
    # look at the upper part of the band only: base plates and brackets at the floor are
    # often wider than the posts and close the gap between two railings down there
    H = info["z1"] - info["z0"]
    pu = pu[pz > info["z0"] + 0.45 * H]
    nb = max(1, int(math.ceil((u1 - u0) / cell)))
    if len(pu) < 8 * nb:                           # too sparse to tell empty from unsampled
        return []
    h, e = np.histogram(pu, bins=nb, range=(u0, u1))
    empty = h == 0
    out, j = [], 0
    while j < nb:
        if empty[j]:
            k = j
            while k < nb and empty[k]:
                k += 1
            # a real gap: the rails run right up to it on both sides (between the bars
            # of a part without a handrail up there it is also empty — that's no gap)
            w = max(3, int(round(0.3 / cell)))
            left, right = ~empty[max(0, j - w):j], ~empty[k:k + w]
            if ((k - j) * cell >= min_gap and j > 0 and k < nb and len(left) >= 3
                    and len(right) >= 3 and left.mean() >= 0.9 and right.mean() >= 0.9):
                out.append((float(e[j]), float(e[k])))
            j = k
        else:
            j += 1
    return out


def cut_gaps(card, info):
    """Remove the card where the railing has a gap, so the texture doesn't bridge it."""
    import bmesh
    gaps = find_gaps(info)
    if not gaps:
        return 0
    U = Vector(info["u"])
    me = card.data
    bm = bmesh.new()
    bm.from_mesh(me)
    for a, b in gaps:
        for x in (a, b):
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bmesh.ops.bisect_plane(bm, geom=geom, plane_co=U * x, plane_no=U)
    dead = [f for f in bm.faces
            if any(a < f.calc_center_median().dot(U) < b for a, b in gaps)]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bm.to_mesh(me)
    bm.free()
    me.update()
    return len(gaps)


def warp_to_profile(card, info):
    """Cut the card at the profile breakpoints and lift it back onto the profile."""
    import bmesh
    bu, bz = info["profile"]
    U = Vector(info["u"])
    me = card.data
    bm = bmesh.new()
    bm.from_mesh(me)
    for x in bu[1:-1]:
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=U * float(x), plane_no=U)
    Z = Vector(info["z"])
    for v in bm.verts:
        v.co += Z * float(np.interp(v.co.dot(U), bu, bz))
    bm.to_mesh(me)
    bm.free()
    me.update()


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _sources(context):
    return [o for o in context.selected_objects if o.type in HIGH_TYPES and o.visible_get()]


def analyse(context, s):
    objs = _sources(context)
    if not objs:
        raise RuntimeError("Выделите геометрию (забор, решётку…)")
    dg = context.evaluated_depsgraph_get()
    pts = gather_points(objs, dg, samples=_SAMPLES["n"])
    if len(pts) == 0:
        raise RuntimeError("У выделенных объектов нет граней")
    u, z, n = card_frame(pts, s.side)
    pu, pz, pn = project(pts, u, z, n)
    # exact extents from vertices, not samples
    allv = []
    for o in objs:
        A = C.mesh_arrays(o, dg)
        allv.append(A.world_co())
    V = np.concatenate(allv)
    vu, vz, vn = project(V, u, z, n)
    info = {
        "objs": objs, "u": u, "z": z, "n": n,
        "u0": float(vu.min()), "u1": float(vu.max()),
        "z0": float(vz.min()), "z1": float(vz.max()),
        "n0": float(vn.min()), "n1": float(vn.max()),
    }
    info["profile"] = None
    prof = top_profile(pu, pz, info["u0"], info["u1"]) if s.follow_profile else None
    if prof is not None:
        bu, bz = prof
        bz = bz - float(bz.min())
        info["profile"] = (bu, bz)
        pz = pz - np.interp(pu, bu, bz)
        vz = vz - np.interp(vu, bu, bz)
        info["z0"], info["z1"] = float(vz.min()), float(vz.max())
    info["length"] = info["u1"] - info["u0"]
    info["height"] = info["z1"] - info["z0"]
    info["samples"] = (pu, pz)
    info["thickness"] = info["n1"] - info["n0"]
    # the card stands where the bars are (middle of the band), not in the middle of the
    # bounding box: a handrail sticking out would put it in front of the bars → parallax
    H_ = info["z1"] - info["z0"]
    mid = (pz > info["z0"] + 0.25 * H_) & (pz < info["z1"] - 0.25 * H_)
    nm = float(np.median(pn[mid])) if mid.sum() > 100 else info.get("n_card", (info["n0"] + info["n1"]) / 2)
    info["n_card"] = min(max(nm, info["n0"]), info["n1"])
    score = 0.0
    info["period_rejected"] = None
    if s.period_mode == "AUTO":
        period = None
        for cand, _sc in find_periods(pu - info["u0"], pz, info["length"])[:12]:
            check = validate_period(pu - info["u0"], pz - info["z0"], info["length"],
                                    info["height"], cand)
            if check >= 0.8:
                period, score = cand, check
                break
            if info["period_rejected"] is None or check > info["period_rejected"][1]:
                info["period_rejected"] = (cand, check)
    elif s.period_mode == "MANUAL":
        period = s.period
    else:
        period = None
    info["period"] = period
    info["score"] = score
    tile = int(s.tile_size)
    info["plan"] = plan(info["length"], info["height"], period, tile, s.model_type,
                        s.density_min, s.density_max, s.quadrant, s.max_tiles)
    if s.model_type == "NPM" and info["plan"]["tiled"]:
        # the quarter is ours to use as we like: bake the object whole (no repeat) when it
        # fits at the minimal density — every bar lands exactly where it is
        pu_ = plan(info["length"], info["height"], None, tile, s.model_type,
                   s.density_min, s.density_max, s.quadrant, 1)
        if pu_["tiles"] == 1 and pu_["density"] >= s.density_min - 1e-6:
            pu_["unique"] = True
            info["plan"] = pu_
    return info


def describe(info):
    p = info["plan"]
    lines = [f"Длина {info['length']:.2f} м, высота {info['height']:.2f} м, "
             f"толщина {info['thickness']:.2f} м"]
    if info["period"]:
        lines.append(f"Пролёт {info['period']:.3f} м"
                     + (f" (уверенность {info['score']:.2f})" if info["score"] else ""))
    else:
        rej = info.get("period_rejected")
        lines.append("Повтор не найден" + (f" (шаг {rej[0]:.2f} м похож, но совпадение "
                                            f"{rej[1]:.2f} < 0.8)" if rej else "")
                     + " — карточка без тайлинга")
    if info.get("profile") is not None:
        lines.append(f"Профиль: {len(info['profile'][0]) - 1} участк(а/ов), "
                     f"перепад {float(np.ptp(info['profile'][1])):.2f} м — выпрямляю для запекания")
    if not p["tiled"] and p["pieces"]:
        lines.append(f"Нарезано на {len(p['pieces'])} полос(ы), тайлов: {p['tiles']}")
    if p["tiled"]:
        lines.append(f"В текстуре {p['spans']} пролёт(а) = {p['seg_len']:.2f} м, "
                     f"повторов ×{p['repeats']:.2f}")
    lines.append(f"Плотность {p['density']:.1f} px/м, занято {p['used_w']}×{p['used_h']} px "
                 f"из {p['area'][2]}×{p['area'][3]}")
    return lines, p["warnings"]


_last = {"lines": [], "warnings": []}


class TK_OT_card_analyse(bpy.types.Operator):
    bl_idname = "tk.card_analyse"
    bl_label = "Проверить"
    bl_description = "Посчитать пролёт, плотность и тайлинг без запекания"

    def execute(self, context):
        try:
            info = analyse(context, context.scene.tk_card)
        except RuntimeError as ex:
            self.report({"WARNING"}, str(ex))
            return {"CANCELLED"}
        lines, warns = describe(info)
        _last.update(lines=lines, warnings=warns)
        self.report({"INFO"}, " · ".join(lines))
        for w in warns:
            self.report({"WARNING"}, w)
        return {"FINISHED"}


class TK_OT_card_make(bpy.types.Operator):
    bl_idname = "tk.card_make"
    bl_label = "Сделать карточку"
    bl_description = ("Запечь выделенную геометрию на плоскость с прозрачностью, затайлить по "
                      "пролёту и заменить геометрию этой плоскостью")
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        n_runs = count_runs(context)
        if n_runs >= 2:
            self.report({"INFO"}, f"В выделении {n_runs} отдельных веток перил — делаю пакетно")
            return bpy.ops.tk.card_batch("INVOKE_DEFAULT")
        context.window.cursor_set("WAIT")
        try:
            self._checked = True
            return self.execute(context)
        finally:
            context.window.cursor_set("DEFAULT")

    def execute(self, context):
        s = context.scene.tk_card
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        n_runs = 0 if getattr(self, "_checked", False) else count_runs(context)
        if n_runs >= 2:
            # several separate railings (floors, sides, flights) in the selection: one card
            # through all of them is nonsense — do each one, sharing textures
            r = bpy.ops.tk.card_batch("EXEC_DEFAULT")
            if r == {"FINISHED"}:
                self.report({"INFO"}, f"В выделении {n_runs} отдельных веток перил — "
                                      "сделано пакетно, по карточке на ветку")
            return r
        try:
            info = analyse(context, s)
        except RuntimeError as ex:
            self.report({"WARNING"}, str(ex))
            return {"CANCELLED"}
        lines, warns = describe(info)
        _last.update(lines=lines, warnings=warns)
        originals = list(info["objs"])
        info["collection"] = (originals[0].users_collection[0] if originals[0].users_collection
                              else context.scene.collection)
        info["name"] = f"{originals[0].name}_card"
        temps = []
        if info["profile"] is not None:
            temps = straightened_copies(context, originals, info)
            info["objs"] = temps
        try:
            res = self._make(context, s, info, lines, warns)
        finally:
            for t in temps:
                me = t.data
                bpy.data.objects.remove(t)
                if me.users == 0:
                    bpy.data.meshes.remove(me)
        if res is None or isinstance(res, set):
            return res
        card = res
        if info["profile"] is not None:
            warp_to_profile(card, info)
        p_ = info["plan"]
        if s.fit_sections and p_["tiled"] and p_["wrap"] and "bar_offset" in info:
            fit_card_uvs(card, info)
        ng = cut_gaps(card, info)
        if ng:
            lines.append(f"Разрывов в перилах: {ng} — карточка там вырезана")
        card["tk_card_density"] = info["plan"]["density"]
        _last_texture["tex"] = info.get("texture") if info["plan"]["tiled"] else None
        return self._finish(context, s, originals, card, lines, warns)

    def _make(self, context, s, info, lines, warns):
        p = info["plan"]
        if not p["tiled"]:
            return self._make_packed(context, s, info, p, lines, warns)
        tile = int(s.tile_size)
        u, z, n = info["u"], info["z"], info["n"]
        objs = info["objs"]
        x0, y0, aw, ah = p["area"]
        used_w, used_h = p["used_w"], p["used_h"]
        seg = p["seg_len"]
        # segment to bake: start where a whole number of periods begins near the middle
        if p["tiled"]:
            s0 = typical_segment(info, seg)
            if s.fit_sections:
                # bake exactly k bars of the local step, window starting between two bars
                pu_, _ = info["samples"]
                Pb, phb, _sc = fit_local_period(pu_, s0 - 2 * info["period"],
                                                s0 + seg + 2 * info["period"], info["period"])
                if phb is not None:
                    info["bars_per_period"] = bars_per_period(
                        pu_, s0 - 2 * Pb, s0 + seg + 2 * Pb, Pb)
                    info["period"] = Pb
                    p = plan(info["length"], info["height"], Pb, tile, s.model_type,
                             s.density_min, s.density_max, s.quadrant, s.max_tiles)
                    info["plan"] = p
                    used_w, used_h = p["used_w"], p["used_h"]
                    seg = p["seg_len"]
                    s0 = phb - Pb / 2 + math.floor((s0 - (phb - Pb / 2)) / Pb) * Pb
                    info["bar_offset"] = 0.5          # bars sit mid-period in the texture
        else:
            s0 = info["u0"]
        z0 = info["z0"]
        z1 = z0 + used_h / p["density"]               # whole pixels in height
        eps = max(1e-4, info["thickness"] * 0.02)
        nc = info["n1"] + eps                          # plane just in front of everything
        extrusion = eps
        max_dist = info["thickness"] + 3 * eps

        # bake plane covering exactly the used pixels of the area
        rx0, ry0 = x0 / tile, y0 / tile
        rx1, ry1 = (x0 + used_w) / tile, (y0 + used_h) / tile
        a1 = s0 + used_w / p["density"]
        corners = [frame_point(u, z, n, s0, z0, nc), frame_point(u, z, n, a1, z0, nc),
                   frame_point(u, z, n, a1, z1, nc), frame_point(u, z, n, s0, z1, nc)]
        pmat = bpy.data.materials.new("TK_card_bake")
        try:
            pmat.use_nodes = True
        except Exception:
            pass
        plane = make_quad_object("TK_card_bake", corners,
                                 [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)],
                                 context.scene.collection, pmat)
        arrays = {}
        try:
            margin = max(4, int(s.bleed_px))
            for ch in s.channels(objs):
                arrays[ch] = bake_segment(context, objs, plane, ch, tile, extrusion, max_dist,
                                          margin, s.ao_samples, aa_samples(p["density"]))
        finally:
            me = plane.data
            bpy.data.objects.remove(plane)
            bpy.data.meshes.remove(me)
            bpy.data.materials.remove(pmat)

        # crop the used rectangle, post-process, write into the atlas
        sl = (slice(y0, y0 + used_h), slice(x0, x0 + used_w))
        crop = {ch: a[sl].copy() for ch, a in arrays.items()}
        if finish_maps(crop, s) < 0.01:
            self.report({"ERROR"}, "Ничего не запеклось — проверьте, что геометрия видна "
                                   "с выбранной стороны")
            return {"CANCELLED"}
        images = write_atlas(s.atlas_name, crop, tile, sl)
        if s.save_files:
            self._save(s, images)

        # the card itself
        mat = card_material(s.atlas_name, images)
        if p["wrap"]:
            uv_rect = (rx0, ry0, rx1, ry1)                   # whole tile: plain repeat
        else:
            inset = 0.5 / tile                               # keep sampling inside the area
            uv_rect = (rx0 + inset, ry0, rx1 - inset, ry1)
        coll = info["collection"]
        name = info["name"]
        vrect = _scale_v(uv_rect, used_h, info["height"], p["density"])
        card = build_card_mesh(name, u, z, n, info["u0"], info["u1"], z0,
                               z0 + info["height"], info.get("n_card", (info["n0"] + info["n1"]) / 2), s0, seg,
                               vrect, p["wrap"], coll, mat)
        info["texture"] = {"P": info["period"], "k": max(1, p["spans"]), "H": info["height"],
                           "m": info.get("bars_per_period", 1),
                           "zprof": z_profile(info, info["z1"], info["height"]),
                           "uv": vrect, "mat": mat, "density": p["density"], "seg": seg,
                           "wrap": p["wrap"], "s0": s0}
        return card

    def _finish(self, context, s, objs, card, lines, warns):
        for o in objs:
            if s.originals == "HIDE":
                o.hide_set(True)
                o.hide_render = True
            elif s.originals == "DELETE":
                bpy.data.objects.remove(o)
        C.deselect_all(context.view_layer)
        card.select_set(True)
        context.view_layer.objects.active = card
        self.report({"INFO"}, "Карточка готова · " + " · ".join(lines))
        for w in warns:
            self.report({"WARNING"}, w)
        return {"FINISHED"}

    def _make_packed(self, context, s, info, p, lines, warns):
        """Non-repeating object: strips packed into one or more texture areas."""
        import bmesh
        tile = int(s.tile_size)
        u, z, n = info["u"], info["z"], info["n"]
        objs = info["objs"]
        x0, y0, aw, ah = p["area"]
        eps = max(1e-4, info["thickness"] * 0.02)
        nc = info["n1"] + eps
        verts, faces, uvs, uvs_in, mat_idx = [], [], [], [], []
        for (ua, ub, za, zb, t, px, py, w, h) in p["pieces"]:
            a, b = info["u0"] + ua, info["u0"] + ub
            c0, c1 = info["z0"] + za, info["z0"] + zb
            j = len(verts)
            for (uu, zz) in ((a, c0), (b, c0), (b, c1), (a, c1)):
                q = frame_point(u, z, n, uu, zz, nc)
                verts.append((q.x, q.y, q.z))
            faces.append((j, j + 1, j + 2, j + 3))
            U0, V0 = (x0 + px) / tile, (y0 + py) / tile
            U1, V1 = (x0 + px + w) / tile, (y0 + py + h) / tile
            uvs += [(U0, V0), (U1, V0), (U1, V1), (U0, V1)]
            # final UVs stay half a pixel inside, so filtering never reads the gutter;
            # the bake itself uses the exact rectangle so edge pixels get baked
            hp = 0.5 / tile
            uvs_in += [(U0 + hp, V0 + hp), (U1 - hp, V0 + hp), (U1 - hp, V1 - hp), (U0 + hp, V1 - hp)]
            mat_idx.append(t)
        me = bpy.data.meshes.new(info["name"])
        me.from_pydata(verts, [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        for i, dta in enumerate(uvl.data):
            dta.uv = uvs[i]
        tmp_mats = []
        for t in range(p["tiles"]):
            tm = bpy.data.materials.new(f"TK_card_bake_{t}")
            try:
                tm.use_nodes = True
            except Exception:
                pass
            me.materials.append(tm)
            tmp_mats.append(tm)
        for poly, t in zip(me.polygons, mat_idx):
            poly.material_index = t
        coll = info["collection"]
        card = bpy.data.objects.new(info["name"], me)
        coll.objects.link(card)

        sl = (slice(y0, y0 + ah), slice(x0, x0 + aw))
        per_tile = [dict() for _ in range(p["tiles"])]
        try:
            margin = max(4, int(s.bleed_px))
            for ch in s.channels(objs):
                arrs = bake_targets(context, objs, card, ch, tile, eps,
                                    info["thickness"] + 3 * eps, margin, s.ao_samples,
                                    aa_samples(p["density"]))
                for t, a in enumerate(arrs):
                    per_tile[t][ch] = a[sl].copy()
        except Exception:
            bpy.data.objects.remove(card)
            bpy.data.meshes.remove(me)
            for tm in tmp_mats:
                bpy.data.materials.remove(tm)
            raise
        filled = max(finish_maps(crop, s) for crop in per_tile)
        if filled < 0.01:
            bpy.data.objects.remove(card)
            bpy.data.meshes.remove(me)
            for tm in tmp_mats:
                bpy.data.materials.remove(tm)
            self.report({"ERROR"}, "Ничего не запеклось — проверьте, что геометрия видна "
                                   "с выбранной стороны")
            return {"CANCELLED"}
        for t, crop in enumerate(per_tile):
            name = s.atlas_name if p["tiles"] == 1 else f"{s.atlas_name}_{t + 1}"
            images = write_atlas(name, crop, tile, sl)
            if s.save_files:
                self._save(s, images)
            me.materials[t] = card_material(name, images)
        for tm in tmp_mats:
            bpy.data.materials.remove(tm)
        uvl = me.uv_layers["UVMap"]
        for i, dta in enumerate(uvl.data):
            dta.uv = uvs_in[i]
        # weld the grid (UV seams stay) and move the card to the middle of the depth
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        shift = n * (info.get("n_card", (info["n0"] + info["n1"]) / 2) - nc)
        for v in bm.verts:
            v.co += shift
        bm.to_mesh(me)
        bm.free()
        return card

    def _save(self, s, images):
        if s.output_dir.startswith("//") and not bpy.data.filepath:
            self.report({"WARNING"}, "Сохраните .blend, чтобы писать текстуры по пути //")
            return
        d = bpy.path.abspath(s.output_dir)
        os.makedirs(d, exist_ok=True)
        for ch, img in images.items():
            path = os.path.join(d, img.name + M.EXT[s.file_format])
            img.filepath_raw = path
            img.file_format = s.file_format
            img.save()


def _scale_v(uv_rect, used_h, height, density):
    """Card spans the true height; the texture holds ceil(height·d) rows — map exactly."""
    x0, y0, x1, y1 = uv_rect
    true_h = height * density
    frac = true_h / used_h if used_h else 1.0
    return (x0, y0, x1, y0 + (y1 - y0) * min(1.0, frac))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class TK_PT_cards(bpy.types.Panel):
    bl_label = "Карточки из геометрии"
    bl_idname = "TK_PT_cards"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    def draw(self, context):
        s = context.scene.tk_card
        layout = self.layout
        layout.label(text="Выделите забор / решётку / ограду", icon="INFO")
        layout.prop(s, "model_type", expand=True)
        layout.prop(s, "tile_size", expand=True)
        r = layout.row(align=True)
        r.prop(s, "density_min")
        r.prop(s, "density_max")
        if s.model_type == "NPM":
            layout.prop(s, "quadrant")
        else:
            layout.prop(s, "max_tiles")
        layout.prop(s, "follow_profile")
        layout.prop(s, "period_mode")
        if s.period_mode == "MANUAL":
            layout.prop(s, "period")
        if s.period_mode != "NONE":
            layout.prop(s, "fit_sections")
        layout.prop(s, "side", expand=True)

        box = layout.box()
        box.label(text="Каналы: Diffuse + Opacity и…", icon="NODE_TEXTURE")
        g = box.row(align=True)
        g.prop(s, "ch_normal", toggle=True)
        g.prop(s, "ch_roughness", toggle=True)
        g.prop(s, "ch_metallic", toggle=True)
        g.prop(s, "ch_ao", toggle=True)
        box.prop(s, "auto_pbr")
        box.prop(s, "bleed")
        if s.bleed:
            box.prop(s, "bleed_px")
        box.prop(s, "threshold")
        box.prop(s, "binary_mask")

        layout.prop(s, "atlas_name")
        layout.prop(s, "originals")
        layout.prop(s, "save_files")
        if s.save_files:
            layout.prop(s, "output_dir")
            layout.prop(s, "file_format")

        r = layout.row(align=True)
        r.operator("tk.card_analyse", icon="VIEWZOOM")
        r.scale_y = 1.3
        r.operator("tk.card_make", icon="IMAGE_PLANE")
        box = layout.box()
        box.label(text="Комплекс перил (много веток)", icon="OUTLINER_OB_GROUP_INSTANCE")
        col = box.column(align=True)
        col.scale_y = 1.2
        col.operator("tk.card_batch", icon="MOD_ARRAY")
        box.prop(s, "batch_rest")
        box.operator("tk.card_repack", icon="UV_SYNC_SELECT")
        if _last_batch:
            box.label(text=f"Веток {_last_batch['runs']}, карточек {_last_batch['cards']}, "
                           f"типов {len(_last_batch.get('types', []))}, "
                           f"не подошло {len(_last_batch['failed'])}")
        if _last["lines"]:
            box = layout.box()
            for ln in _last["lines"]:
                box.label(text=ln)
            for w in _last["warnings"]:
                box.label(text=w, icon="ERROR")


# ---------------------------------------------------------------------------
# Batch: a whole railing complex → runs → cards sharing one texture
# ---------------------------------------------------------------------------

def z_profile(info, ztop, H, nb=40):
    """Where along the height the material is (rails, bars), normalised, top-aligned."""
    _pu, pz = info["samples"]
    h, _ = np.histogram(ztop - pz, bins=nb, range=(0.0, H))
    h = h.astype(float)
    return h / max(h.sum(), 1e-9)


def _rails(h, k=5):
    """Horizontal members only: peaks of the height profile above its local background
    (balusters give a flat background whose level depends on their spacing)."""
    pad = np.pad(h, k, mode="edge")
    bg = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(h))])
    r = np.clip(h - bg, 0, None)
    return r / max(r.max(), 1e-9)


def profile_match(a, b):
    ra, rb = _rails(a), _rails(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def card_with_texture(context, s, obj, tex, name, coll, info=None):
    """Card for one run *obj* that reuses an already baked texture *tex*.
    Returns (card, info) or (None, reason)."""
    if info is None:
        info = analyse(context, s)
    H = tex["H"]
    if abs(info["height"] - H) > 0.05 * H:
        return None, f"другая высота ({info['height']:.2f} м против {H:.2f} м)"
    if info["length"] < 0.5 * tex["P"]:
        return None, "слишком короткий"
    zm = profile_match(z_profile(info, info["z1"], H), tex["zprof"])
    if zm < 0.6:
        return None, f"другое устройство по высоте (сходство {zm:.2f})"
    info["bars_per_period"] = tex["m"]
    u, z, n = info["u"], info["z"], info["n"]
    z1 = info["z1"]
    Hc = min(H, info["height"])                     # a lower railing crops, never stretches
    z0 = z1 - Hc                                    # align the handrail (top of the band)
    k = tex["k"]
    ux0, uy0, ux1, uy1 = tex["uv"]
    vrect = (ux0, uy1 - (uy1 - uy0) * Hc / H, ux1, uy1)
    card = build_card_mesh(name, u, z, n, info["u0"], info["u1"], z0, z1,
                           info.get("n_card", (info["n0"] + info["n1"]) / 2), info["u0"], tex["seg"],
                           vrect, True, coll, tex["mat"])
    if info["profile"] is not None:
        warp_to_profile(card, info)
    info["plan"]["spans"] = k
    info["bar_offset"] = 0.5
    if tex.get("tb") is not None:
        info["texture"] = {"tb": tex["tb"], "m": tex.get("m", 1)}   # measured bar positions
    secs = fit_sections(info, tex["P"], tex["m"])
    good = [x for x in secs if x["score"] > 2.0]
    if not good:
        bpy.data.meshes.remove(card.data)
        return None, "шаг стоек не совпадает с текстурой"
    info["period"] = tex["P"]
    fit_card_uvs(card, info)
    cut_gaps(card, info)
    return card, info


def count_runs(context, min_length=1.0):
    """How many separate railing runs ≥ min_length the selected meshes hold."""
    from . import runs as RU
    n = 0
    for src in [o for o in context.selected_objects if o.type == "MESH"]:
        me = src.data
        nv = len(me.vertices)
        if nv == 0:
            continue
        co = np.empty(nv * 3)
        me.vertices.foreach_get("co", co)
        mw = np.array([list(r) for r in src.matrix_world])
        W = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        try:
            _lab, _parts, _lines, runs, _left = RU.analyse_mesh(W, me)
        except Exception:
            continue
        n += sum(1 for r in runs if r["u1"] - r["u0"] >= min_length)
    return n


class TK_OT_card_batch(bpy.types.Operator):
    bl_idname = "tk.card_batch"
    bl_label = "Перила целиком"
    bl_description = ("Разобрать выделенный комплекс перил на отдельные ветки, запечь одну общую "
                      "текстуру с самой длинной ветки и поставить карточки на все ветки")
    bl_options = {"REGISTER", "UNDO"}

    min_length: FloatProperty(name="Короче — оставить геометрией", default=1.0, min=0.0,
                              subtype="DISTANCE",
                              description="Ветки короче остаются геометрией. Если не задано: "
                                          "ВПМ — 1 м, НПМ — 0,25 м (НПМ запекает ветку целиком, "
                                          "ей повтор не нужен)")
    join: BoolProperty(name="Собрать карточки в один объект", default=True)
    max_types: bpy.props.IntProperty(name="Максимум типов перил (текстур)", default=6, min=1,
                                     max=16)

    # The work is a generator: execute() runs it to the end, invoke() runs it from a timer
    # one step at a time, so Blender keeps redrawing, shows progress and Esc cancels.
    def _steps(self, context):
        from . import runs as RU
        s = context.scene.tk_card
        self._result = {"CANCELLED"}
        self._temp, self._made, self._srcs = [], [], []
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        srcs = [o for o in context.selected_objects if o.type == "MESH"]
        if not srcs:
            self.report({"WARNING"}, "Выделите меш с перилами")
            return
        wm = context.window_manager
        coll = srcs[0].users_collection[0] if srcs[0].users_collection else context.scene.collection
        run_objs, leftovers = [], []
        self._srcs = list(srcs)
        min_len = (self.min_length if self.properties.is_property_set("min_length")
                   else (0.25 if s.model_type == "NPM" else 1.0))
        for si, src in enumerate(srcs):
            yield 0.02, f"Разбор меша {src.name} на ветки…"
            me = src.data
            mcache = RU.mesh_cache(me)
            nv = len(me.vertices)
            co = np.empty(nv * 3)
            me.vertices.foreach_get("co", co)
            mw = np.array([list(r) for r in src.matrix_world])
            W = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
            lab, parts, lines, runs, left = RU.analyse_mesh(W, me)
            npoly = len(me.polygons)
            ls_ = np.empty(npoly, np.int64)
            me.polygons.foreach_get("loop_start", ls_)
            lv = np.empty(len(me.loops), np.int64)
            me.loops.foreach_get("vertex_index", lv)
            plab = lab[lv[ls_]]
            part_run = np.array([p.run for p in parts])
            part_line = np.array([p.line for p in parts])
            # polygon centres (world) and materials: a handrail bent around a corner is ONE
            # part but belongs to two runs — each run also takes the polygons of other
            # parts that lie in its own plane
            lt_ = np.empty(npoly, np.int64)
            me.polygons.foreach_get("loop_total", lt_)
            pc = np.add.reduceat(W[lv], ls_, axis=0) / lt_[:, None]
            pmi = np.empty(npoly, np.int64)
            me.polygons.foreach_get("material_index", pmi)
            keep = np.zeros(len(parts), bool)        # parts that stay geometry
            for i, r in enumerate(runs):
                if r["u1"] - r["u0"] < min_len:
                    keep[[p.idx for p in r["parts"]]] = True
                    continue
                own = part_run[plab] == i
                d2, n2 = np.asarray(r["dir"]), np.asarray(r["nrm"])
                ownc = pc[own]
                off = float(np.median(ownc[:, :2] @ n2)) if len(ownc) else 0.0
                uu = pc[:, :2] @ d2
                band = ((np.abs(pc[:, :2] @ n2 - off) < 0.06)
                        & (uu > r["u0"] - 0.05) & (uu < r["u1"] + 0.05)
                        & (pc[:, 2] > r["z0"] - 0.2) & (pc[:, 2] < r["z1"] + 0.3))
                if len(ownc):
                    mats = np.unique(pmi[own])
                    band &= np.isin(pmi, mats)
                # only parts of OTHER lines (a bent handrail); railings of other floors
                # on this same line stay theirs
                band &= part_line[plab] != r.get("line", -2)
                # …and only at the height of this run's own geometry at that spot (a sloped
                # run's bounding box would reach the floors above and below)
                if len(ownc) and band.any():
                    ou = ownc[:, :2] @ d2
                    b0 = float(r["u0"]) - 0.1
                    nb_ = int((float(r["u1"]) + 0.1 - b0) / 0.25) + 1
                    ob = np.clip(((ou - b0) / 0.25).astype(int), 0, nb_ - 1)
                    zlo = np.full(nb_, np.inf)
                    zhi = np.full(nb_, -np.inf)
                    np.minimum.at(zlo, ob, ownc[:, 2])
                    np.maximum.at(zhi, ob, ownc[:, 2])
                    cand = np.nonzero(band)[0]
                    cb = np.clip(((uu[cand] - b0) / 0.25).astype(int), 0, nb_ - 1)
                    lo_ = np.minimum(zlo[cb], zlo[np.clip(cb - 1, 0, nb_ - 1)])
                    hi_ = np.maximum(zhi[cb], zhi[np.clip(cb + 1, 0, nb_ - 1)])
                    ok = (pc[cand, 2] > lo_ - 0.1) & (pc[cand, 2] < hi_ + 0.15)
                    band[cand[~ok]] = False
                sub = RU.submesh(me, own | band, f"{src.name}_run{i}", mcache)
                o = bpy.data.objects.new(f"{src.name}_run{i}", sub)
                o.matrix_world = src.matrix_world
                context.scene.collection.objects.link(o)
                run_objs.append((o, r, src, i))
                self._temp.append(o)
            for p in left:
                keep[p.idx] = True
            leftovers.append((src, keep[plab] if len(keep) else None, part_run, plab, mcache))
        if not run_objs:
            self.report({"WARNING"}, "Не нашлось веток длиннее порога")
            return

        def select_only(o):
            C.deselect_all(context.view_layer)
            o.hide_set(False)
            o.select_set(True)
            context.view_layer.objects.active = o

        # quick look at every run: height and where the rails are
        run_objs.sort(key=lambda t: -(t[1]["u1"] - t[1]["u0"]))
        for o, *_ in run_objs:
            o.hide_set(True)
        prev = (s.originals, s.period_mode, s.atlas_name)
        quick = {}
        s.period_mode = "NONE"
        try:
            for j, (o, *_r) in enumerate(run_objs):
                yield 0.05 + 0.15 * j / len(run_objs), f"Обмер веток {j + 1}/{len(run_objs)}"
                select_only(o)
                inf = analyse(context, s)
                o.hide_set(True)
                # a railing band is not taller than ~2.5 m; taller means the profile
                # wasn't found (or it isn't a railing) — such runs stay geometry
                quick[o.name] = (inf["height"], inf, inf["height"] < 2.5)
        finally:
            s.period_mode = prev[1]

        groups = []
        if s.model_type == "NPM":
            cards, failed = yield from self._npm_steps(context, s, run_objs, quick, coll,
                                                       prev, groups)
        else:
            # group runs into railing types; the longest run of a type is its master
            groups = []                     # dicts: master, tex, members
            cards, failed = [], []
            s.originals = "KEEP"
            s.period_mode = "AUTO"
            try:
                for j, (o, r, src, i) in enumerate(run_objs):
                    frac = 0.2 + 0.75 * j / len(run_objs)
                    yield frac, f"Карточки: ветка {j + 1}/{len(run_objs)}, типов {len(groups)}"
                    H, inf, plausible = quick[o.name]
                    home, best = None, 0.0
                    for g in groups:
                        gH = g["H"]
                        if abs(H - gH) > 0.03 * gH:
                            continue
                        mt = profile_match(z_profile(inf, inf["z1"], gH), g["tex"]["zprof"])
                        if mt >= 0.7 and mt > best:
                            home, best = g, mt
                    if home is not None:
                        select_only(o)
                        card, why = card_with_texture(context, s, o, home["tex"],
                                                      f"{src.name}_card{i}", coll, info=inf)
                        o.hide_set(True)
                        if card is not None:
                            cards.append(card)
                            self._made.append(card)
                            home["n"] += 1
                            continue
                        failed.append((o.name, why))
                        continue
                    if not plausible or len(groups) >= self.max_types:
                        failed.append((o.name, "высота больше 2,5 м — профиль не распознан" if not plausible
                                       else "слишком много типов перил"))
                        continue
                    # a new railing type: bake it
                    gname = s.atlas_name if not groups else f"{prev[2]}_{len(groups) + 1}"
                    yield frac, (f"Запекание текстуры типа {len(groups) + 1} "
                                 f"(ветка {j + 1}/{len(run_objs)}, {r['u1'] - r['u0']:.1f} м)…")
                    s.atlas_name = gname
                    select_only(o)
                    res = bpy.ops.tk.card_make()
                    mc = context.view_layer.objects.active
                    o.hide_set(True)
                    tex = _last_texture.get("tex")
                    if "FINISHED" not in res or mc is o or tex is None:
                        if "FINISHED" in res and mc is not o and mc is not None:
                            bpy.data.objects.remove(mc)
                        failed.append((o.name, "не тайлится (нет повторяющихся стоек)"))
                        continue
                    cards.append(mc)
                    self._made.append(mc)
                    groups.append({"H": tex["H"], "tex": tex, "n": 1, "atlas": gname})
            finally:
                s.originals, s.period_mode, s.atlas_name = prev
        yield 0.96, "Сборка результата…"
        if not cards:
            self.cleanup()
            self.report({"ERROR"}, "Ни одна ветка не превратилась в карточку")
            return

        # leftovers: parts not converted (short runs, unassigned, failed runs) stay as geometry
        failed_ids = {}
        for oname, why in failed:
            src_name = oname.split("_run")[0]
            failed_ids.setdefault(src_name, []).append(int(oname.rsplit("_run", 1)[1]))
        rest_objs = []
        for src, keepmask, part_run, plab, mcache in leftovers:
            mask = keepmask.copy() if keepmask is not None else np.zeros(len(src.data.polygons), bool)
            for ri in failed_ids.get(src.name, []):
                mask |= part_run[plab] == ri
            drop = s.batch_rest == "DELETE" or (s.batch_rest == "AUTO" and s.model_type == "NPM")
            if mask.any() and drop:
                self._dropped = getattr(self, "_dropped", 0) + int(mask.sum())
            elif mask.any():
                me2 = RU.submesh(src.data, mask, f"{src.name}_остаток", mcache)
                ro = bpy.data.objects.new(f"{src.name}_остаток", me2)
                ro.matrix_world = src.matrix_world
                coll.objects.link(ro)
                rest_objs.append(ro)
            src.hide_set(True)
            src.hide_render = True
        self._remove_temp()

        for c in cards:
            if c.name not in coll.objects:
                for uc in list(c.users_collection):
                    uc.objects.unlink(c)
                coll.objects.link(c)
        result = cards
        if self.join and len(cards) > 1:
            C.deselect_all(context.view_layer)
            for c in cards:
                c.select_set(True)
            context.view_layer.objects.active = cards[0]
            with context.temp_override(active_object=cards[0], selected_editable_objects=cards,
                                       selected_objects=cards):
                bpy.ops.object.join()
            joined = context.view_layer.objects.active
            joined.name = f"{srcs[0].name}_карточки"
            result = [joined]
        C.deselect_all(context.view_layer)
        for c in result + rest_objs:
            c.select_set(True)
        context.view_layer.objects.active = result[0]
        tri = sum(len(c.data.polygons) for c in result)
        rest = sum(len(c.data.polygons) for c in rest_objs)
        dropped = getattr(self, "_dropped", 0)
        self._dropped = 0
        _last_batch.update(runs=len(run_objs), cards=len(cards), failed=list(failed),
                           card_polys=tri, rest_polys=rest, dropped_polys=dropped,
                           types=[(g["atlas"], g["n"], round(g["H"], 2)) for g in groups])
        msg = (f"Веток: {len(run_objs)}, карточек: {len(cards)}, типов перил: {len(groups)}, "
               f"полигонов в карточках: {tri}; "
               f"оставлено геометрией: {rest} полигонов")
        if dropped:
            msg += f"; мелкие остатки убраны ({dropped} полигонов, исходник скрыт в сцене)"
        if failed:
            msg += f"; не подошли {len(failed)}" + ("" if dropped else " (оставлены геометрией)")
        self.report({"INFO"}, msg)
        self._result = {"FINISHED"}


    def _npm_steps(self, context, s, run_objs, quick, coll, prev, groups):
        """НПМ: no repeats — every run is baked whole, and the strips of ALL runs are
        packed into the chosen quarter of one texture (UVs never leave it)."""
        tile = int(s.tile_size)
        area = area_rect(tile, "NPM", s.quadrant)
        x0, y0, aw, ah = area
        cards, failed = [], []
        items = []
        for o, r, src, i in run_objs:
            H, inf, plausible = quick[o.name]
            if not plausible:
                failed.append((o.name, "высота больше 2,5 м — профиль не распознан"))
                continue
            items.append((o.name, inf["length"], inf["height"]))
        # all at the best common density; if even the minimum doesn't fit, the shortest
        # runs stay geometry until the rest fits
        d, placed = best_run_packing(items, aw, ah, s.density_min, s.density_max)
        dropped = []
        while d is None and items:
            items.sort(key=lambda t: t[1] * t[2])
            dropped.append(items.pop(0))
            d, placed = best_run_packing(items, aw, ah, s.density_min, s.density_max)
        for key, *_ in dropped:
            failed.append((key, f"не влезло в четверть при {s.density_min:g} px/м"))
        if not items:
            return cards, failed
        groups.append({"atlas": s.atlas_name, "n": 0, "H": max(t[2] for t in items)})
        by_name = {o.name: (o, r, src, i) for o, r, src, i in run_objs}
        # 1) every run: its card (final) and its quads for the bake target
        per_run = []                      # (key, verts, faces, uv_area, (obj, info), polys)
        acc = None
        try:
            for j, (key, _L, _H) in enumerate(items):
                o, r, src, i = by_name[key]
                if j % 4 == 0:
                    yield (0.2 + 0.2 * j / len(items),
                           f"НПМ {d:.1f} px/м: раскладка веток в четверть {j + 1}/{len(items)}")
                _H, inf, _p = quick[key]
                verts, faces, uvf, uva = piece_geometry(inf, placed[key], area, tile, d)
                eps = max(1e-4, inf["thickness"] * 0.02)
                shift = np.array(inf["n"]) * (inf.get("n_card", (inf["n0"] + inf["n1"]) / 2)
                                              - (inf["n1"] + eps))
                me = mesh_from(f"{src.name}_card{i}", [tuple(np.array(v) + shift) for v in verts],
                               faces, uvf)
                card = bpy.data.objects.new(me.name, me)
                coll.objects.link(card)
                if inf["profile"] is not None:
                    warp_to_profile(card, inf)
                per_run.append((key, verts, faces, uva, (o, inf), len(o.data.polygons)))
                cards.append(card)
                self._made.append(card)
                card["tk_card_density"] = d
                groups[0]["n"] += 1
            # 2) bake many runs at once (Cycles casts against every selected object
            # separately, so they are joined into one), in chunks that fit in memory
            LIMIT = 1_200_000
            chunks, cur, npo = [], [], 0
            for pr in per_run:
                if cur and npo + pr[5] > LIMIT:
                    chunks.append(cur)
                    cur, npo = [], 0
                cur.append(pr)
                npo += pr[5]
            if cur:
                chunks.append(cur)
            g = GUTTER // 2
            for ci, chunk in enumerate(chunks):
                tv, tf, tuv = [], [], []
                # every run is moved far from the others for the bake: one bake pass,
                # but rays from a card only ever reach its own railing (neighbours that
                # pass close by — a stair next to a fence — don't get baked in)
                offs = [(0.0, 0.0, 10000.0 * (q + 1)) for q in range(len(chunk))]
                for q, (key, verts, faces, uva, _src, _n) in enumerate(chunk):
                    k0 = len(tv)
                    oz = offs[q][2]
                    verts = [(v[0], v[1], v[2] + oz) for v in verts]
                    tv += verts
                    tf += [tuple(k0 + q for q in f) for f in faces]
                    tuv += uva
                part = f" (часть {ci + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                yield 0.42 + 0.45 * ci / len(chunks), f"НПМ: сборка веток для запекания{part}…"
                joined = joined_copy(context, [pr[4] for pr in chunk], offsets=offs)
                for pr in chunk:                     # their geometry is in the joined copy
                    o_ = pr[4][0]
                    try:
                        me_ = o_.data
                        bpy.data.objects.remove(o_)
                        if me_.users == 0:
                            bpy.data.meshes.remove(me_)
                    except ReferenceError:
                        pass
                tme = mesh_from("TK_npm_bake", tv, tf, tuv)
                tmat = bpy.data.materials.new("TK_card_bake")
                try:
                    tmat.use_nodes = True
                except Exception:
                    pass
                tme.materials.append(tmat)
                target = bpy.data.objects.new("TK_npm_bake", tme)
                coll.objects.link(target)
                try:
                    gen = bake_area(context, s, [joined], target, area, d, thick_of(chunk))
                    while True:
                        try:
                            ch = next(gen)
                        except StopIteration as st:
                            crop = st.value
                            break
                        yield (0.45 + 0.45 * ci / len(chunks),
                               f"НПМ: запекание «{ch}» сразу для {len(chunk)} веток{part}…")
                finally:
                    bpy.data.objects.remove(target)
                    bpy.data.meshes.remove(tme)
                    bpy.data.materials.remove(tmat)
                    jm = joined.data
                    bpy.data.objects.remove(joined)
                    bpy.data.meshes.remove(jm)
                if acc is None:
                    acc = {ch: np.zeros_like(a) for ch, a in crop.items()}
                for pr in chunk:
                    for (_ua, _ub, _za, _zb, _t, px, py, w, h) in placed[pr[0]]:
                        ys = slice(max(0, py - g), min(ah, py + h + g))
                        xs = slice(max(0, px - g), min(aw, px + w + g))
                        for ch, a in crop.items():
                            if ch in acc:
                                acc[ch][ys, xs] = a[ys, xs]
                crop = None
        finally:
            pass
        if not acc:
            return cards, failed
        yield 0.92, "НПМ: доводка карт и запись атласа…"
        finish_maps(acc, s)
        sl = (slice(y0, y0 + ah), slice(x0, x0 + aw))
        images = write_atlas(s.atlas_name, acc, tile, sl)
        mat = card_material(s.atlas_name, images)
        for c in cards:
            c.data.materials.clear()
            c.data.materials.append(mat)
            c["tk_card_density"] = d
        self._npm_density = d
        return cards, failed

    def _remove_temp(self):
        for o in self._temp:
            try:
                me_ = o.data
                bpy.data.objects.remove(o)
                if me_.users == 0:
                    bpy.data.meshes.remove(me_)
            except ReferenceError:
                pass
        self._temp = []

    def cleanup(self):
        """Undo what a cancelled / failed run left behind."""
        self._remove_temp()
        for c in self._made:
            try:
                me_ = c.data
                bpy.data.objects.remove(c)
                if me_ is not None and me_.users == 0:
                    bpy.data.meshes.remove(me_)
            except ReferenceError:
                pass
        self._made = []
        for src in self._srcs:
            try:
                src.hide_set(False)
                src.select_set(True)
            except (ReferenceError, RuntimeError):
                pass

    def execute(self, context):
        dbg = os.environ.get("TK_DBG")
        try:
            for frac, text in self._steps(context):
                if dbg:
                    print(f"STEP {int(frac * 100)}% {text}", flush=True)
        except Exception:
            self.cleanup()
            raise
        return self._result

    def invoke(self, context, event):
        self._gen = self._steps(context)
        self._busy = False
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        wm.progress_begin(0, 1000)
        self._status(context, "Перила целиком: начинаю…")
        return {"RUNNING_MODAL"}

    def _status(self, context, text):
        try:
            context.workspace.status_text_set(text + "   ·   Esc — отменить")
        except Exception:
            pass
        if context.area:
            context.area.header_text_set(text)
        C.tag_redraw_3d()

    def _end(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass
        if context.area:
            context.area.header_text_set(None)

    def modal(self, context, event):
        if event.type == "ESC":
            self._end(context)
            s_ = context.scene.tk_card
            self._gen.close()                  # runs the generator's finally blocks
            self.cleanup()
            self.report({"WARNING"}, "Отменено — сцена возвращена как была")
            return {"CANCELLED"}
        if event.type != "TIMER" or self._busy:
            return {"PASS_THROUGH"}
        self._busy = True
        try:
            frac, text = next(self._gen)
        except StopIteration:
            self._end(context)
            return self._result
        except Exception as ex:
            self._end(context)
            self.cleanup()
            self.report({"ERROR"}, f"Ошибка: {ex}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        finally:
            self._busy = False
        context.window_manager.progress_update(int(frac * 1000))
        self._status(context, f"Перила целиком: {int(frac * 100)}% · {text}")
        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
# Repack: several finished (non-repeating) cards into ONE quarter, no rebake
# ---------------------------------------------------------------------------

def _resample_axis(a, n, axis):
    """Exact area-average resampling of *a* to length *n* along *axis*."""
    a = np.moveaxis(a, axis, 0)
    m = a.shape[0]
    c = np.concatenate([np.zeros((1,) + a.shape[1:], a.dtype), np.cumsum(a, axis=0)], 0)
    e = np.linspace(0.0, m, n + 1)
    i = np.minimum(np.floor(e).astype(int), m - 1)
    f = (e - i)[(...,) + (None,) * (a.ndim - 1)]
    C = c[i] + (c[i + 1] - c[i]) * f                  # integral up to each edge
    out = (C[1:] - C[:-1]) / (e[1:] - e[:-1])[(...,) + (None,) * (a.ndim - 1)]
    return np.moveaxis(out, 0, axis)


def resample(a, h, w):
    return _resample_axis(_resample_axis(a, h, 0), w, 1)


def card_islands(ob):
    """UV islands of a card (pieces): list of dicts with faces, slot, uv bbox. Raises
    ValueError for cards whose UVs leave 0…1 (tiled ВПМ cards repeat the texture)."""
    me = ob.data
    uvl = me.uv_layers.active
    if uvl is None:
        raise ValueError("нет UV")
    nl = len(me.loops)
    uv = np.empty(nl * 2, np.float32)
    uvl.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    if uv.size and (uv.min() < -1e-4 or uv.max() > 1.0 + 1e-4):
        raise ValueError("карточка с повтором текстуры (ВПМ) — её не переупаковать, "
                         "только перезапечь в НПМ")
    faces = []
    for p in me.polygons:
        li = list(p.loop_indices)
        b = uv[li]
        faces.append((p.index, p.material_index, b.min(0), b.max(0)))
    # faces whose UV boxes touch belong to one island (pieces are ≥ GUTTER px apart)
    n = len(faces)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    eps = 1e-5
    order = sorted(range(n), key=lambda k: faces[k][2][0])
    for a_i, ia in enumerate(order):
        fa = faces[ia]
        for ib in order[a_i + 1:]:
            fb = faces[ib]
            if fb[2][0] > fa[3][0] + eps:
                break
            if fa[1] == fb[1] and fb[2][1] <= fa[3][1] + eps and fa[2][1] <= fb[3][1] + eps:
                par[find(ia)] = find(ib)
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    out = []
    for ks in groups.values():
        lo = np.min([faces[k][2] for k in ks], 0)
        hi = np.max([faces[k][3] for k in ks], 0)
        out.append({"faces": [faces[k][0] for k in ks], "slot": faces[ks[0]][1],
                    "lo": lo, "hi": hi})
    return out, uv


def slot_images(ob, slot):
    """{channel: Image} of a card material slot (atlas images named <atlas>_<Channel>)."""
    if slot >= len(ob.material_slots) or ob.material_slots[slot].material is None:
        return {}
    mat = ob.material_slots[slot].material
    out = {}
    if not mat.node_tree:
        return out
    for nd in mat.node_tree.nodes:
        if nd.bl_idname == "ShaderNodeTexImage" and nd.image:
            for ch, spec in M.CHANNELS.items():
                if nd.image.name.endswith(spec[1]):
                    out[ch] = nd.image
    return out


def island_density(ob, isl, uv, tile_px):
    """px per metre of an island, measured on its faces (UV length / 3D length)."""
    me = ob.data
    mw = ob.matrix_world
    r = []
    for fi in isl["faces"]:
        p = me.polygons[fi]
        li = list(p.loop_indices)
        vs = [mw @ me.vertices[me.loops[l].vertex_index].co for l in li]
        for k in range(len(li)):
            a, b = li[k], li[(k + 1) % len(li)]
            L = (vs[(k + 1) % len(li)] - vs[k]).length
            if L > 1e-3:
                r.append(float(np.linalg.norm((uv[b] - uv[a]) * tile_px)) / L)
    return float(np.median(r)) if r else 0.0


def repack_cards(context, objs, s, op=None):
    """Move the texture pieces of *objs* (finished non-repeating cards) into one quarter
    of the atlas s.atlas_name at one common density. Returns (density, n_islands)."""
    tile = int(s.tile_size)
    area = area_rect(tile, "NPM", s.quadrant)
    x0, y0, aw, ah = area
    items = []                     # (ob, island, images, src_tile, px rect, density)
    for ob in objs:
        isls, uv = card_islands(ob)
        for isl in isls:
            imgs = slot_images(ob, isl["slot"])
            if "OPACITY" not in imgs and "DIFFUSE" not in imgs:
                raise ValueError(f"{ob.name}: не нашёл текстур карточки в материале")
            st = next(iter(imgs.values())).size[0]
            rx0 = int(math.floor(isl["lo"][0] * st + 1e-4))
            ry0 = int(math.floor(isl["lo"][1] * st + 1e-4))
            rx1 = int(math.ceil(isl["hi"][0] * st - 1e-4))
            ry1 = int(math.ceil(isl["hi"][1] * st - 1e-4))
            rx1, ry1 = max(rx1, rx0 + 1), max(ry1, ry0 + 1)
            d = island_density(ob, isl, uv, st)
            items.append((ob, isl, imgs, st, (rx0, ry0, rx1, ry1), d, uv))
    if not items:
        raise ValueError("нечего паковать")
    d_src_min = min(it[5] for it in items if it[5] > 0)
    d_hi = min(s.density_max, d_src_min)          # never stretch above the source
    if d_hi < s.density_min:
        d_hi = s.density_min

    def layout(d):
        sizes = []
        for it in items:
            rx0, ry0, rx1, ry1 = it[4]
            k = d / it[5] if it[5] > 0 else 1.0
            sizes.append((max(1, math.ceil((rx1 - rx0) * k)), max(1, math.ceil((ry1 - ry0) * k))))
        order = sorted(range(len(items)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
        pos = [None] * len(items)
        x = y = sh = 0
        for i in order:
            w, h = sizes[i]
            if x + w > aw:
                x, y, sh = 0, y + sh + GUTTER, 0
            if w > aw or y + h > ah:
                return None, sizes
            pos[i] = (x, y)
            x += w + GUTTER
            sh = max(sh, h)
        return pos, sizes

    d = d_hi
    pos = None
    while d >= s.density_min - 1e-6:
        pos, sizes = layout(d)
        if pos is not None:
            break
        d *= 0.97
    if pos is None:
        need = sum(w * h for w, h in layout(s.density_min)[1])
        raise ValueError(f"не влезает в четверть {aw}×{ah} даже при {s.density_min:g} px/м "
                         f"(нужно ~{need / (aw * ah) * 100:.0f}% площади четверти) — возьмите "
                         f"тайл 4096 или разнесите карточки по двум четвертям")
    # compose the quarter
    cache = {}

    def arr(img):
        if img.name not in cache:
            cache[img.name] = M.image_array(img)
        return cache[img.name]
    chans = sorted({ch for it in items for ch in it[2]}, key=M.ORDER.index)
    acc = {}
    for ch in chans:
        a = np.zeros((ah, aw, 4), np.float32)
        if ch != "OPACITY":
            a[..., :3] = M.DEFAULT_FILL[ch]
        a[..., 3] = 1.0 if ch != "DIFFUSE" else 0.0
        acc[ch] = a
    for it, (px, py), (w, h) in zip(items, pos, sizes):
        ob, isl, imgs, st, (rx0, ry0, rx1, ry1), d_src, uv = it
        op = None
        if "OPACITY" in imgs:
            op = resample(arr(imgs["OPACITY"])[ry0:ry1, rx0:rx1], h, w)
            acc["OPACITY"][py:py + h, px:px + w] = op
        for ch in chans:
            if ch == "OPACITY" or ch not in imgs:
                continue
            src = arr(imgs[ch])[ry0:ry1, rx0:rx1]
            if op is not None:
                wsrc = arr(imgs["OPACITY"])[ry0:ry1, rx0:rx1][..., :1]
                num = resample(src * wsrc, h, w)
                den = resample(wsrc, h, w)
                blk = np.where(den > 1e-6, num / np.maximum(den, 1e-6), resample(src, h, w))
            else:
                blk = resample(src, h, w)
            acc[ch][py:py + h, px:px + w] = blk
        # UVs: same piece, new place and scale
        sx, sy = w / (rx1 - rx0), h / (ry1 - ry0)
        me = ob.data
        for fi in isl["faces"]:
            for li in me.polygons[fi].loop_indices:
                u_px, v_px = uv[li][0] * st, uv[li][1] * st
                uv[li] = ((x0 + px + (u_px - rx0) * sx) / tile,
                          (y0 + py + (v_px - ry0) * sy) / tile)
    finish_maps(acc, s)
    sl = (slice(y0, y0 + ah), slice(x0, x0 + aw))
    images = write_atlas(s.atlas_name, acc, tile, sl)
    if s.save_files and op is not None:
        TK_OT_card_make._save(op, s, images)
    mat = card_material(s.atlas_name, images)
    done = set()
    for it in items:
        ob = it[0]
        if ob.name in done:
            continue
        done.add(ob.name)
        me = ob.data
        me.uv_layers.active.data.foreach_set("uv", it[6].ravel())
        mi = np.zeros(len(me.polygons), np.int32)
        me.polygons.foreach_set("material_index", mi)
        me.materials.clear()
        me.materials.append(mat)
        me.update()
        ob["tk_card_density"] = d
    return d, len(items)


class TK_OT_card_repack(bpy.types.Operator):
    bl_idname = "tk.card_repack"
    bl_label = "Упаковать в одну четверть"
    bl_description = ("Собрать текстуры выделенных готовых карточек в выбранную четверть "
                      "одного атласа (без перезапекания): одна плотность в пределах «от–до», "
                      "UV и материалы меняются, геометрия карточек — нет")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.tk_card
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        if not objs:
            self.report({"WARNING"}, "Выделите готовые карточки")
            return {"CANCELLED"}
        try:
            d, n = repack_cards(context, objs, s, self)
        except ValueError as ex:
            self.report({"ERROR"}, f"Не упаковалось: {ex}")
            return {"CANCELLED"}
        q = s.bl_rna.properties["quadrant"].enum_items[s.quadrant].name
        self.report({"INFO"}, f"Упаковано {n} кусков из {len(objs)} карточек в четверть «{q}» "
                              f"атласа «{s.atlas_name}», {d:.1f} px/м")
        return {"FINISHED"}


_last_texture = {}
_last_batch = {}
_SAMPLES = {"n": 200000}


classes = (TK_CardSettings, TK_OT_card_analyse, TK_OT_card_make, TK_OT_card_batch,
           TK_OT_card_repack, TK_PT_cards)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_card = bpy.props.PointerProperty(type=TK_CardSettings)


def unregister():
    del bpy.types.Scene.tk_card
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


