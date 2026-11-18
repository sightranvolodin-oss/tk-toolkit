# Two cards baked separately (different atlases / quarters) → "Упаковать в одну четверть":
# all UVs in the chosen quarter, one material, the pattern (posts) unchanged.
import os, sys, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bpy, bmesh, numpy as np
from mathutils import Matrix, Vector
import tk_toolkit as TK
TK.register()
from tk_toolkit import cards as K
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc = bpy.context.scene
def fence(name, x0, y, L, step, h):
    bm = bmesh.new()
    def tube(a, b, r):
        a, b = Vector(a), Vector(b); d = b - a
        m = Matrix.Translation((a + b) / 2) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=r, radius2=r, depth=d.length, matrix=m)
    tube((x0, y, h), (x0 + L, y, h), 0.03); tube((x0, y, 0.1), (x0 + L, y, 0.1), 0.02)
    for i in range(int(L / step) + 1):
        tube((x0 + i * step, y, 0.1), (x0 + i * step, y, h), 0.02)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    m = bpy.data.materials.new(name + "_mat"); m.use_nodes = True
    next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled").inputs["Base Color"].default_value = (0.8, 0.1, 0.1, 1)
    me.materials.append(m)
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); return ob
A = fence("FenceA", 0, 0, 30.0, 0.5, 1.0)
B = fence("FenceB", 0, 5, 18.0, 0.3, 1.2)
s = sc.tk_card
s.model_type = "NPM"; s.density_min, s.density_max = K.TYPE_DEFAULTS["NPM"]; s.save_files = False
cards = []
for ob, atl, q in ((A, "AtlA", "LT"), (B, "AtlB", "RT")):
    for x in bpy.data.objects: x.select_set(False)
    ob.hide_set(False); ob.select_set(True); bpy.context.view_layer.objects.active = ob
    s.atlas_name, s.quadrant = atl, q
    r = bpy.ops.tk.card_make(); c = bpy.context.view_layer.objects.active
    print(ob.name, r, c.name, len(c.data.polygons), c.get("tk_card_density"))
    assert r == {"FINISHED"} and c is not ob
    cards.append(c)

def profile(c):
    """opacity along the middle of the strip(s), sampled through the card's own UVs"""
    me = c.data; uvl = me.uv_layers.active
    img = next(n.image for n in c.material_slots[0].material.node_tree.nodes
               if n.bl_idname == "ShaderNodeTexImage" and n.image and n.image.name.endswith("_Opacity"))
    a = K.M.image_array(img)[..., 0]; T = a.shape[0]
    out = []
    for p in sorted(me.polygons, key=lambda p: (p.center.x, p.center.z)):
        uv = np.array([uvl.data[l].uv[:] for l in p.loop_indices]); co = np.array([me.vertices[me.loops[l].vertex_index].co[:] for l in p.loop_indices])
        lo, hi = uv.min(0), uv.max(0)
        # world x ↔ u: sample 0.6 of height, step 2 cm
        x0, x1 = co[:, 0].min(), co[:, 0].max()
        for x in np.arange(x0, x1, 0.02):
            t = (x - x0) / max(x1 - x0, 1e-9)
            U = lo[0] + t * (hi[0] - lo[0]); V = lo[1] + 0.6 * (hi[1] - lo[1])
            out.append(a[min(T - 1, int(V * T)), min(T - 1, int(U * T))])
    return np.array(out)

before = [profile(c) for c in cards]
for x in bpy.data.objects: x.select_set(False)
for c in cards: c.hide_set(False); c.select_set(True)
bpy.context.view_layer.objects.active = cards[0]
s.atlas_name, s.quadrant = "Packed", "RB"
r = bpy.ops.tk.card_repack(); print("repack", r)
assert r == {"FINISHED"}
for c in cards:
    uv = np.array([d.uv[:] for d in c.data.uv_layers.active.data])
    print(c.name, "uv", uv.min(0).round(3), uv.max(0).round(3), "density", round(c["tk_card_density"], 2), [m.name for m in c.data.materials])
    assert uv[:, 0].min() >= 0.5 and uv[:, 0].max() <= 1.0 and uv[:, 1].min() >= 0 and uv[:, 1].max() <= 0.5
    assert len(c.data.materials) == 1 and c.data.materials[0].name.startswith("Packed")
    assert 10 <= c["tk_card_density"] <= 40
after = [profile(c) for c in cards]
for b, a in zip(before, after):
    # posts at the same places: binarised profiles agree
    n = min(len(a), len(b)); bb, aa = b[:n] > 0.3, a[:n] > 0.3
    agree = (bb == aa).mean(); print("pattern agreement", round(float(agree), 3), "posts before/after", int(np.sum(np.diff(bb.astype(int)) == 1)), int(np.sum(np.diff(aa.astype(int)) == 1)))
    assert agree > 0.9
# too much for the quarter → a clear error, nothing changed
s.density_min = 39.0; s.density_max = 40.0; s.tile_size = "2048"
big = fence("Huge", 0, 10, 1500.0, 5.0, 1.0)
for x in bpy.data.objects: x.select_set(False)
big.select_set(True); bpy.context.view_layer.objects.active = big; s.atlas_name, s.quadrant = "AtlH", "LB"
s.density_min = 10.0
bpy.ops.tk.card_make(); hc = bpy.context.view_layer.objects.active
for x in bpy.data.objects: x.select_set(False)
for c in cards + [hc]: c.hide_set(False); c.select_set(True)
s.density_min = 39.0
uv0 = np.array([d.uv[:] for d in cards[0].data.uv_layers.active.data])
try:
    r = bpy.ops.tk.card_repack()
except RuntimeError as ex:
    r = str(ex)
print("too big:", r)
uv1 = np.array([d.uv[:] for d in cards[0].data.uv_layers.active.data])
assert np.allclose(uv0, uv1)
