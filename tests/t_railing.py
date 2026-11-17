import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import math, bpy, bmesh, numpy as np
from mathutils import Vector
import tk_toolkit as TK
TK.register()
from tk_toolkit import cards as K
OUT = os.environ.get("TK_TEST_OUT", "")
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc = bpy.context.scene
PROF = [(0, 0), (3, 0), (7, 2.4), (10, 2.4), (13, 4.2), (20, 4.2)]   # (x, floor z)
def floor_z(x): return float(np.interp(x, [p[0] for p in PROF], [p[1] for p in PROF]))
def mat(name, col, metal=0.0):
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except Exception: pass
    b = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Metallic"].default_value = metal
    return m
bm = bmesh.new()
def beam(a, b, t, mi=0):
    """square bar from a to b, thickness t"""
    a, b = Vector(a), Vector(b); d = b - a; L = d.length
    r = bmesh.ops.create_cube(bm, size=1.0)
    rot = d.to_track_quat('X', 'Z').to_matrix()
    for v in r["verts"]:
        v.co = a + rot @ Vector((v.co.x * L + L / 2, v.co.y * t, v.co.z * t))
for i in range(len(PROF) - 1):
    (x0, z0), (x1, z1) = PROF[i], PROF[i + 1]
    beam((x0, 0, z0 + 1.0), (x1, 0, z1 + 1.0), 0.05)    # handrail
    beam((x0, 0, z0 + 0.12), (x1, 0, z1 + 0.12), 0.035)  # bottom rail
for x, z in PROF:
    beam((x, 0, z), (x, 0, z + 1.08), 0.07)              # posts at breaks
x = 0.06
while x < 20:
    z = floor_z(x)
    beam((x, 0, z + 0.12), (x, 0, z + 1.0), 0.022)      # balusters, 0.12 m horizontally
    x += 0.12
me = bpy.data.meshes.new("Railing"); bm.to_mesh(me); bm.free()
rail = bpy.data.objects.new("Railing", me); sc.collection.objects.link(rail)
me.materials.append(mat("Steel", (0.75, 0.2, 0.05), 0.4))
rail.rotation_euler.z = math.radians(15)
# stairs under it (not selected)
bm = bmesh.new()
for i in range(len(PROF) - 1):
    (x0, z0), (x1, z1) = PROF[i], PROF[i + 1]
    nst = 1 if z0 == z1 else int(abs(z1 - z0) / 0.18)
    for k in range(nst):
        xa = x0 + (x1 - x0) * k / nst; xb = x0 + (x1 - x0) * (k + 1) / nst
        zt = z0 + (z1 - z0) * (k + 1) / nst if z1 != z0 else z0
        r = bmesh.ops.create_cube(bm, size=1.0)
        for v in r["verts"]:
            v.co = Vector(((v.co.x + 0.5) * (xb - xa) + xa, v.co.y * 1.2 + 0.65, (v.co.z - 0.5) * 0.15 + zt))
me = bpy.data.meshes.new("Stairs"); bm.to_mesh(me); bm.free()
st = bpy.data.objects.new("Stairs", me); sc.collection.objects.link(st); st.rotation_euler.z = math.radians(15)
me.materials.append(mat("Concrete", (0.5, 0.5, 0.52)))
bpy.context.view_layer.update()

s = sc.tk_card
for o in bpy.data.objects: o.select_set(False)
rail.select_set(True); bpy.context.view_layer.objects.active = rail
info = K.analyse(bpy.context, s)
lines, warns = K.describe(info)
print("ANALYSE:", " · ".join(lines), warns)
bu, bz = info["profile"]
print("profile breakpoints u:", np.round(bu - bu[0], 2), "z:", np.round(bz, 2))

cam_d = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
R = rail.matrix_world.to_3x3()
mid = rail.matrix_world @ Vector((10, 0, 2.6))
cam.location = mid + R @ Vector((-1.5, -15.5, 1.2)); cam.rotation_euler = (mid - cam.location).to_track_quat('-Z', 'Y').to_euler(); cam_d.lens = 32
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.rotation_euler = (math.radians(50), 0, math.radians(-20)); sun.data.energy = 3.5
w = bpy.data.worlds.new("w"); sc.world = w
try: w.use_nodes = True
except Exception: pass
next(n for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeBackground").inputs[0].default_value = (0.35, 0.4, 0.45, 1)
sc.render.engine = "CYCLES"; sc.cycles.samples = 24; sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 760, 340
def render(name):
    path = os.path.join(OUT or "/tmp", name); sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(path); a = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4); bpy.data.images.remove(im); return a
st.hide_render = True                                     # silhouette comparison without the stairs
sc.render.film_transparent = True
ref = render("rail_geom_alpha.png")
results = []
for label, mt in (("ВПМ 2048", "VPM"), ("НПМ 2048, четверть", "NPM")):
    rail.hide_set(False); rail.hide_render = False
    for o in bpy.data.objects:
        if o.name.startswith("Railing_card"): o.hide_set(True); o.hide_render = True
    s.model_type = mt; s.density_min, s.density_max = K.TYPE_DEFAULTS[mt]; s.tile_size = "2048"; s.atlas_name = "Rail_" + mt
    for o in bpy.data.objects: o.select_set(False)
    rail.select_set(True); bpy.context.view_layer.objects.active = rail
    r = bpy.ops.tk.card_make(); card = bpy.context.view_layer.objects.active
    got = render(f"rail_{mt}_alpha.png")
    ia, ib = ref[..., 3] > 0.5, got[..., 3] > 0.5
    def dil(m, r=2):
        o = m.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                o |= np.roll(np.roll(m, dy, 0), dx, 1)
        return o
    # silhouette agreement with a 2 px tolerance (bars are only 1–9 px wide)
    iou = min((ib & dil(ia)).sum() / max(1, ib.sum()), (ia & dil(ib)).sum() / max(1, ia.sum()))
    # card vertices must follow the handrail profile
    co = np.array([card.matrix_world @ v.co for v in card.data.vertices])
    print(label, r, "quads", len(card.data.polygons), f"совпадение силуэта ±2px {iou:.3f}", "|", " · ".join(K._last["lines"][1:]), K._last["warnings"])
    results.append((mt, iou, len(card.data.polygons)))
assert results[0][1] > 0.85 and results[1][1] > 0.6, results
# pretty renders with stairs for the user
if OUT:
    st.hide_render = False; sc.render.film_transparent = False
    for vis, name in (("geo", "rail_geometry.png"), ("VPM", "rail_card_vpm.png"), ("NPM", "rail_card_npm.png")):
        rail.hide_render = vis != "geo"
        for o in bpy.data.objects:
            if o.name.startswith("Railing_card"):
                o.hide_render = not ((vis == "VPM" and o.data.materials[0].name.startswith("Rail_VPM")) or (vis == "NPM" and o.data.materials[0].name.startswith("Rail_NPM")))
        render(name)
print("railing OK")
