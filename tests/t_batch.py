# Batch railing → cards: one mesh holding two separate straight railings (bars and rails are
# loose parts), plus a stray box. Expect two runs → cards of one type; the box stays as geometry.
import os, sys, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bpy, bmesh, numpy as np
from mathutils import Matrix, Vector
import tk_toolkit as TK
TK.register()
from tk_toolkit import cards as K, runs as R
for o in list(bpy.data.objects): bpy.data.objects.remove(o)

bm = bmesh.new()
def tube(a, b, r=0.02):
    a, b = Vector(a), Vector(b); d = b - a
    m = Matrix.Translation((a + b) / 2) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=r, radius2=r, depth=d.length, matrix=m)
def railing(x0, y, L, step=0.15):
    tube((x0, y, 1.0), (x0 + L, y, 1.0), 0.03)
    tube((x0, y, 0.1), (x0 + L, y, 0.1))
    n = int(L / step)
    for i in range(n + 1):
        tube((x0 + i * step, y, 0.1), (x0 + i * step, y, 1.0), 0.012)
railing(0, 0, 6.0)
railing(0, 3, 4.5)
bmesh.ops.create_cube(bm, size=0.5, matrix=Matrix.Translation((10, 10, 0.25)))
me = bpy.data.meshes.new("Complex"); bm.to_mesh(me); bm.free()
mat = bpy.data.materials.new("Metal"); mat.use_nodes = True; me.materials.append(mat)
ob = bpy.data.objects.new("Complex", me); bpy.context.scene.collection.objects.link(ob)

lab, nparts = R.mesh_loose_parts(me)
print("loose parts", nparts)
assert nparts == 77, nparts  # 2 rails + 41 bars, 2 rails + 31 bars, 1 box

for x in bpy.data.objects: x.select_set(False)
ob.select_set(True); bpy.context.view_layer.objects.active = ob
s = bpy.context.scene.tk_card; s.model_type = "VPM"; s.density_min, s.density_max = K.TYPE_DEFAULTS["VPM"]
s.tile_size = "2048"; s.save_files = False
r = bpy.ops.tk.card_batch(min_length=1.0)
print("batch", r, K._last_batch)
assert r == {"FINISHED"}
b = K._last_batch
assert b["runs"] == 2, b
assert b["cards"] == 2, b
assert len(b["types"]) == 1, b["types"]
names = {o.name for o in bpy.data.objects}
print(sorted(names))
cards = [o for o in bpy.data.objects if o.name.endswith("_карточки")]
rest = [o for o in bpy.data.objects if o.name.endswith("_остаток")]
assert len(cards) == 1 and len(rest) == 1
assert len(cards[0].data.polygons) < 200
assert len(rest[0].data.polygons) == 6, len(rest[0].data.polygons)
img = cards[0].data.materials[0].node_tree.nodes
assert any(n.bl_idname == "ShaderNodeTexImage" and n.image for n in img)
xs = [v.co.x for v in cards[0].data.vertices]
print("card x range", min(xs), max(xs))
assert min(xs) > -0.2 and max(xs) < 6.2
