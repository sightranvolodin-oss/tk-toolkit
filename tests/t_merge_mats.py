# 6 slots: Red, Red.001, Red.002 (same content), Blue, Green, Red-ish (roughness differs)
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bpy, bmesh, numpy as np
from mathutils import Matrix
import tk_toolkit as TK
TK.register()
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
def mat(name, col, rough):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = col; b.inputs["Roughness"].default_value = rough
    return m
ms = [mat("Red", (1,0,0,1), .5), mat("Red.001", (1,0,0,1), .5), mat("Red.002", (1,0,0,1), .5),
      mat("Blue", (0,0,1,1), .5), mat("Green", (0,1,0,1), .5), mat("RedRough", (1,0,0,1), .9)]
bm = bmesh.new()
for i in range(6):
    bmesh.ops.create_cube(bm, size=0.5, matrix=Matrix.Translation((i, 0, 0)))
me = bpy.data.meshes.new("m"); bm.to_mesh(me); bm.free()
for m in ms: me.materials.append(m)
for p in me.polygons: p.material_index = p.index // 6
ob = bpy.data.objects.new("Obj", me); bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob; ob.select_set(True)
r = bpy.ops.tk.split_merge_same(); print("merge", r, [m.name for m in me.materials])
assert [m.name for m in me.materials] == ["Red", "Blue", "Green", "RedRough"]
mi = np.array([p.material_index for p in me.polygons])
assert (mi[:18] == 0).all() and (mi[18:24] == 1).all() and (mi[30:] == 3).all()
assert ob.active_material_index == 0
# and pull it out
r = bpy.ops.tk.split_merge_same(extract=True); print("again", r)   # nothing more to merge
bpy.context.scene.tk_split.mode = "ACTIVE"; bpy.context.scene.tk_split.extract = "CUT"
ob.active_material_index = 0
r = bpy.ops.tk.split_by_material(); print("split", r, [(o.name, len(o.data.polygons)) for o in bpy.data.objects])
assert any(len(o.data.polygons) == 18 for o in bpy.data.objects if o is not ob)
# extract=True: merge and pull out the biggest set in one go
me2 = ob.data
me2.materials.append(mat("Blue.001", (0,0,1,1), .5)); me2.materials.append(mat("Blue.002", (0,0,1,1), .5))
for p in me2.polygons:
    if p.index >= 12: p.material_index = len(me2.materials) - 1 - (p.index % 2)
for o in bpy.data.objects: o.select_set(False)
ob.select_set(True); bpy.context.view_layer.objects.active = ob
r = bpy.ops.tk.split_merge_same(extract=True); print("extract", r, [(o.name, len(o.data.polygons)) for o in bpy.data.objects])
assert any(o.name.startswith("Obj_Blue") for o in bpy.data.objects)
