# Two materials, islands already laid out in one UV: left half has a textured material,
# right half a plain one → "Один атлас по текущей UV" keeps the islands and bakes both.
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bpy, numpy as np
import tk_toolkit as TK
TK.register()
from tk_toolkit import maps as M
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
me = bpy.data.meshes.new("m")
me.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(2,0,0),(3,0,0),(3,1,0),(2,1,0)], [], [(0,1,2,3),(4,5,6,7)])
uv = me.uv_layers.new(name="UVMap")
uv.data.foreach_set("uv", np.array([0,0, .5,0, .5,1, 0,1,  .5,0, 1,0, 1,1, .5,1], np.float32))
img = bpy.data.images.new("src", 64, 64); px = np.zeros((64,64,4), np.float32); px[..., 3] = 1
px[:, :, 0] = np.linspace(0, 1, 64)[None, :]          # red ramp along U
img.pixels.foreach_set(px.ravel())
A = bpy.data.materials.new("A"); A.use_nodes = True; nt = A.node_tree
bs = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = img; nt.links.new(tx.outputs["Color"], bs.inputs["Base Color"])
bs.inputs["Roughness"].default_value = 0.9
B = bpy.data.materials.new("B"); B.use_nodes = True
bb = next(n for n in B.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
bb.inputs["Base Color"].default_value = (0, 1, 0, 1); bb.inputs["Roughness"].default_value = 0.2
me.materials.append(A); me.materials.append(B); me.polygons[1].material_index = 1
ob = bpy.data.objects.new("Obj", me); bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob; ob.select_set(True)
p = bpy.context.scene.tk_maps
p.ch_roughness = True; p.atlas_size = "512"; p.atlas_on_copy = False; p.output_dir = "/tmp/tk_atlas_uv/"
r = bpy.ops.tk.maps_atlas_uv(); print("op", r)
assert r == {"FINISHED"}
assert len(me.materials) == 1
mat = me.materials[0]
imgs = {n.image.name: n.image for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeTexImage" and n.image}
print(sorted(imgs))
D = M.image_array(next(i for k, i in imgs.items() if k.endswith("_Diffuse")))
R = M.image_array(next(i for k, i in imgs.items() if k.endswith("_Roughness")))
row = D[256]
print("left ramp", row[32, :3].round(2), row[224, :3].round(2), "right", row[384, :3].round(2))
assert row[384, 1] > 0.9 and row[384, 0] < 0.1               # plain green island
assert row[224, 0] > row[32, 0] + 0.3                        # textured ramp kept in place
print("rough L/R", R[256, 128, 0].round(2), R[256, 384, 0].round(2))
assert abs(R[256, 128, 0] - 0.9) < 0.05 and abs(R[256, 384, 0] - 0.2) < 0.05
