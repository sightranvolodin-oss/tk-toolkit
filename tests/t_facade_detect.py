import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy, bmesh, numpy as np
import tk_toolkit as TK
TK.register()
from tk_toolkit import cards as K
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc = bpy.context.scene
rng = np.random.default_rng(4)
def mat(name, col, metal=0.0, rough=0.4):
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except Exception: pass
    b = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Metallic"].default_value = metal; b.inputs["Roughness"].default_value = rough
    return m
# --- panels: staggered rows, varying length, 3 depth layers, rounded
bm = bmesh.new()
L = 36.0; rows = 6; row_h = 1.35
for r in range(rows):
    x = -rng.uniform(0, 2.0) if r % 2 else 0.0
    while x < L:
        w = rng.uniform(2.6, 4.2)
        if rng.random() > 0.12:
            depth = [0.0, 0.35, 0.7][rng.integers(0, 3)]
            res = bmesh.ops.create_cube(bm, size=1.0)
            vs = res["verts"]
            for v in vs:
                v.co.x = v.co.x * (w - 0.15) + x + w / 2
                v.co.y = v.co.y * 0.4 - depth
                v.co.z = v.co.z * (row_h * 0.75) + 1.0 + r * row_h + rng.uniform(-0.05, 0.05) * 0
        x += w
for v in bm.verts: v.co.x = min(max(v.co.x, 0.0), L)
me = bpy.data.meshes.new("Panels"); bm.to_mesh(me); bm.free()
pan = bpy.data.objects.new("Panels", me); sc.collection.objects.link(pan)
bev = pan.modifiers.new("bevel", "BEVEL"); bev.width = 0.18; bev.segments = 5
pan.data.materials.append(mat("PanelMetal", (0.25, 0.26, 0.28), 0.6, 0.35))
for p in pan.data.polygons: p.use_smooth = True
# --- behind: red frame, stairs, floors (NOT selected)
bm = bmesh.new()
def box(c, s):
    res = bmesh.ops.create_cube(bm, size=1.0)
    for v in res["verts"]:
        v.co.x = v.co.x * s[0] + c[0]; v.co.y = v.co.y * s[1] + c[1]; v.co.z = v.co.z * s[2] + c[2]
for i in range(13): box((i * 3, 2.0, 5), (0.2, 0.2, 10))
for f in range(4): box((L / 2, 2.5, f * 3 + 0.1), (L, 3, 0.15))
for st in range(30): box((5 + st * 0.3, 2.5, st * 0.2), (0.3, 1.0, 0.05))
me = bpy.data.meshes.new("Frame"); bm.to_mesh(me); bm.free()
frame = bpy.data.objects.new("Frame", me); sc.collection.objects.link(frame)
frame.data.materials.append(mat("Red", (0.35, 0.03, 0.04)))
wall = bpy.data.objects.new("BackWall", bpy.data.meshes.new("w")); sc.collection.objects.link(wall)
wall.data.from_pydata([(0, 4.5, 0), (L, 4.5, 0), (L, 4.5, 10), (0, 4.5, 10)], [], [(0, 1, 2, 3)])
wall.data.materials.append(mat("Concrete", (0.6, 0.6, 0.6)))
bpy.context.view_layer.update()
for o in bpy.data.objects: o.select_set(False)
pan.select_set(True); bpy.context.view_layer.objects.active = pan
s = sc.tk_card
info = K.analyse(bpy.context, s)
print(" · ".join(K.describe(info)[0]))
# a brick-like facade must not get a fake period, nor be taken for a stair railing
assert info["period"] is None, info["period"]
assert info["profile"] is None
p = info["plan"]
assert not p["tiled"] and p["pieces"] and p["used_h"] > 1500   # strips fill the tile
s.max_tiles = 6
p6 = K.analyse(bpy.context, s)["plan"]
assert p6["tiles"] > 1 and p6["density"] > 2 * p["density"]
need = K.tiles_needed(info["length"], info["height"], 2048, 2048, 512)
print("tiles for 512 px/m:", need)
assert need >= 15
print("facade detection OK")
