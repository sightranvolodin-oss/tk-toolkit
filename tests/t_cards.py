import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import math, bpy, bmesh, numpy as np
import tk_toolkit as TK
TK.register()
from tk_toolkit import cards as K
OUT = os.environ.get("TK_TEST_OUT", "")
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc = bpy.context.scene

def mat(name, col):
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except Exception: pass
    b = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*col, 1)
    return m

def fence(length=40.0, span=2.5, rot=30):
    bm = bmesh.new()
    def box(cx, cz, sx, sy, sz, mi):
        r = bmesh.ops.create_cube(bm, size=1.0)
        for v in r["verts"]:
            v.co.x = v.co.x * sx + cx; v.co.y *= sy; v.co.z = v.co.z * sz + cz
        for f in {f for v in r["verts"] for f in v.link_faces}: f.material_index = mi
    n = int(length / span)
    for i in range(n + 1):
        box(i * span, 0.8, 0.12, 0.12, 1.6, 0)                     # posts
    for i in range(n):
        for j in range(9):
            box(i * span + 0.25 * (j + 1), 0.65, 0.08, 0.03, 1.2 - 0.04 * (j % 3), 1)  # pickets
    for zr in (0.3, 1.05):
        box(length / 2, zr, length, 0.04, 0.08, 2)                 # rails
    me = bpy.data.meshes.new("Fence"); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new("Fence", me); sc.collection.objects.link(ob)
    for m in (mat("Post", (0.15, 0.08, 0.04)), mat("Picket", (0.75, 0.6, 0.4)), mat("Rail", (0.4, 0.25, 0.12))):
        me.materials.append(m)
    ob.rotation_euler.z = math.radians(rot); ob.location = (-15, -5, 0)
    return ob

ob = fence()
bpy.context.view_layer.update()
s = sc.tk_card
for o in bpy.data.objects: o.select_set(False)
ob.select_set(True); bpy.context.view_layer.objects.active = ob
info = K.analyse(bpy.context, s)
print("period", round(info["period"] or 0, 4), "score", round(info["score"], 2),
      "len", round(info["length"], 3), "h", round(info["height"], 3), "u", tuple(round(x, 3) for x in info["u"]))
for mt, tile, q in (("VPM", 2048, "LT"), ("VPM", 4096, "LT"), ("NPM", 2048, "RB")):
    p = K.plan(info["length"], info["height"], info["period"], tile, mt, *K.TYPE_DEFAULTS[mt], q)
    print(f"plan {mt} {tile}: d={p['density']:.1f} spans={p['spans']} seg={p['seg_len']:.2f} rep={p['repeats']:.2f} area={p['area']} used={p['used_w']}x{p['used_h']} wrap={p['wrap']} warn={p['warnings']}")

# --- render helper (Cycles, transparent film) ---
cam_d = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam)
sc.camera = cam
cam.location = (0.5, -16, 2.2); cam.rotation_euler = (math.radians(86), 0, math.radians(8))
cam_d.lens = 30
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.rotation_euler = (math.radians(50), 0, math.radians(20)); sun.data.energy = 3
sc.render.engine = "CYCLES"; sc.cycles.samples = 16; sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 640, 300
sc.render.film_transparent = True
def render(path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(path); a = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4); bpy.data.images.remove(im); return a
base = OUT or "/tmp"
ref = render(os.path.join(base, "fence_geometry.png"))

s.model_type = "VPM"; s.tile_size = "2048"; s.period_mode = "AUTO"; s.originals = "HIDE"; s.ch_normal = True
for o in bpy.data.objects: o.select_set(False)
ob.select_set(True); bpy.context.view_layer.objects.active = ob
print("make", bpy.ops.tk.card_make())
card = bpy.context.view_layer.objects.active
print("card", card.name, "faces", len(card.data.polygons), "verts", len(card.data.vertices), "fence hidden", ob.hide_get())
got = render(os.path.join(base, "fence_card.png"))
ia, ib = ref[..., 3] > 0.5, got[..., 3] > 0.5
iou = (ia & ib).sum() / max(1, (ia | ib).sum())
both = ia & ib
cdiff = np.abs(ref[..., :3][both] - got[..., :3][both]).mean()
print(f"silhouette IoU {iou:.3f}  colour diff on overlap {cdiff:.3f}  coverage ref {ia.mean():.3f} card {ib.mean():.3f}")
img = bpy.data.images["Cards_Opacity"]; a = K.M.image_array(img)
print("opacity filled fraction", round(float((a[..., 0] > 0.5).mean()), 3))
# seam check: texture column 0 vs last column of the area should continue the pattern
d = K.M.image_array(bpy.data.images["Cards_Diffuse"])
# NPM quarter on the same atlas name must leave the VPM data alone? use other atlas
ob.hide_set(False); card.hide_set(True)
s.model_type = "NPM"; s.density_min, s.density_max = K.TYPE_DEFAULTS["NPM"]; s.quadrant = "RB"; s.atlas_name = "CardsFar"
for o in bpy.data.objects: o.select_set(False)
ob.select_set(True); bpy.context.view_layer.objects.active = ob
print("make NPM", bpy.ops.tk.card_make())
far = bpy.context.view_layer.objects.active
a = K.M.image_array(bpy.data.images["CardsFar_Opacity"])
h = a.shape[0] // 2
print("NPM quadrant fill  LT", round(float(a[h:, :h, 0].mean()), 3), "RT", round(float(a[h:, h:, 0].mean()), 3),
      "LB", round(float(a[:h, :h, 0].mean()), 3), "RB", round(float(a[:h, h:, 0].mean()), 3))
uv = np.array([d_.uv[:] for d_ in far.data.uv_layers[0].data])
print("NPM uv range", uv.min(0).round(3), uv.max(0).round(3), "quads", len(far.data.polygons))
got2 = render(os.path.join(base, "fence_card_npm.png"))
ib2 = got2[..., 3] > 0.5
print(f"NPM silhouette IoU {((ia & ib2).sum() / max(1, (ia | ib2).sum())):.3f}")

# ---- assertions (CI)
assert info["period"] and abs(info["period"] - 2.5) < 0.02, info["period"]
pv = K.plan(info["length"], info["height"], info["period"], 2048, "VPM", 512, 1702)
assert pv["tiled"] and pv["used_w"] == 2048 and 512 <= pv["density"] <= 1702
pn = K.plan(info["length"], info["height"], info["period"], 2048, "NPM", 10, 40, "RB")
assert pn["area"] == (1024, 0, 1024, 1024) and 10 <= pn["density"] <= 40
assert iou > 0.75, iou
assert len(card.data.polygons) == 1
print("cards assertions OK")
