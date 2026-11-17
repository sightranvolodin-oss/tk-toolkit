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
    binary_mask: BoolProperty(name="Жёсткая маска (0/1)", default=False)

    atlas_name: StringProperty(name="Атлас", default="Cards",
                               description="Имя набора текстур; карточки с одним именем делят тайл")
    originals: EnumProperty(name="Исходная геометрия", items=[
        ("HIDE", "Скрыть", ""), ("DELETE", "Удалить", ""), ("KEEP", "Оставить", "")],
        default="HIDE")
    save_files: BoolProperty(name="Сохранить текстуры в папку", default=False)
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
    bh = float(np.median(band))
    need = max(0.1, 0.25 * bh)
    if bh <= 0 or float(np.ptp(top)) < need or float(np.ptp(bot)) < need:
        return None                            # practically flat
    if float(np.percentile(np.abs(band - bh), 90)) > 0.1 * bh:
        return None                            # the band doesn't keep its height
    dt, db = top - top.mean(), bot - bot.mean()
    if dt.std() == 0 or db.std() == 0 or float(np.corrcoef(dt, db)[0, 1]) < 0.9:
        return None                            # top and bottom don't rise together
    xs = np.concatenate([[u0], xs, [u1]])
    top = np.concatenate([[top[0]], top, [top[-1]]])
    return _douglas_peucker(xs, top, tol)


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


def bake_targets(context, sources, target, ch, tile, extrusion, max_dist, margin, ao_samples):
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
        with M._SceneForBake(context, target, int(ao_samples) if kind == "AO" else 1,
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


def bake_segment(context, sources, plane, ch, tile, extrusion, max_dist, margin, ao_samples):
    return bake_targets(context, sources, plane, ch, tile, extrusion, max_dist, margin,
                        ao_samples)[0]


def finish_maps(crop, s):
    """Flora-style post-processing on one area (dict ch -> array). Returns opacity max."""
    op = crop["OPACITY"][..., 0]
    if s.bleed and "DIFFUSE" in crop:
        d = crop["DIFFUSE"]
        d[..., :3] = M.bleed_colors(d[..., :3], op >= s.threshold, int(s.bleed_px))
    if s.binary_mask:
        op = (op >= s.threshold).astype(np.float32)
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
    med = np.median(W, axis=0)
    dev = np.abs(W - med).sum(axis=(1, 2))
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
    pts = gather_points(objs, dg)
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

    def execute(self, context):
        s = context.scene.tk_card
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
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
        card["tk_card_density"] = info["plan"]["density"]
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
                                          margin, s.ao_samples)
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
        card = build_card_mesh(name, u, z, n, info["u0"], info["u1"], z0,
                               z0 + info["height"], (info["n0"] + info["n1"]) / 2, s0, seg,
                               _scale_v(uv_rect, used_h, info["height"], p["density"]),
                               p["wrap"], coll, mat)
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
                                    info["thickness"] + 3 * eps, margin, s.ao_samples)
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
        shift = n * ((info["n0"] + info["n1"]) / 2 - nc)
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
        if _last["lines"]:
            box = layout.box()
            for ln in _last["lines"]:
                box.label(text=ln)
            for w in _last["warnings"]:
                box.label(text=w, icon="ERROR")


classes = (TK_CardSettings, TK_OT_card_analyse, TK_OT_card_make, TK_PT_cards)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.tk_card = bpy.props.PointerProperty(type=TK_CardSettings)


def unregister():
    del bpy.types.Scene.tk_card
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
