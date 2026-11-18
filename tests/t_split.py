import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, bpy, tempfile
sys.path.insert(0, ROOT)
from tk_toolkit import split as S
S.register()
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc=bpy.context.scene; s=sc.tk_split
def setup():
    bpy.ops.mesh.primitive_cube_add(size=2, location=(3,0,0)); ob=bpy.context.object; ob.name="Box"
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.subdivide(number_cuts=2); bpy.ops.object.mode_set(mode="OBJECT")
    ms=[bpy.data.materials.get(n) or bpy.data.materials.new(n) for n in ("Wood","Metal","Glass")]
    for m in ms: ob.data.materials.append(m)
    ob.data.materials.append(None)
    for p in ob.data.polygons:
        p.material_index = 0 if p.center.z>0.5 else (1 if p.center.x>0.5 else (2 if p.center.y>0.5 else 3))
    return ob, ms
def report(tag):
    objs=sorted((o.name, len(o.data.polygons), [m.name if m else None for m in o.data.materials], tuple(round(x,2) for x in o.location), [c.name for c in o.users_collection], o.hide_get()) for o in bpy.data.objects if o.type=="MESH")
    print(tag); [print("   ",x) for x in objs]
ob,ms=setup(); total=len(ob.data.polygons); print("total faces", total)
s.mode="ALL"; s.on_copy=False; s.origin="BOUNDS"
print(bpy.ops.tk.split_by_material()); report("ALL:")
print("faces sum", sum(len(o.data.polygons) for o in bpy.data.objects))
# join back
part=bpy.data.objects["Box_Wood"]; [o.select_set(False) for o in bpy.data.objects]; part.select_set(True); bpy.context.view_layer.objects.active=part
print(bpy.ops.tk.split_join()); report("JOIN:")
j=bpy.context.view_layer.objects.active; print("joined verts", len(j.data.vertices), "(cube subdiv 2 has 98)", "faces", len(j.data.polygons))
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
for c in list(bpy.data.collections): bpy.data.collections.remove(c)
# PICKED cut + copy
ob,ms=setup()
ms[1].tk_split_pick=True; ms[2].tk_split_pick=True; ms[0].tk_split_pick=False
s.mode="PICKED"; s.extract="CUT"
print(bpy.ops.tk.split_by_material()); report("PICKED CUT (Metal, Glass):")
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
ob,ms=setup(); s.extract="COPY"; s.on_copy=True; s.mode="ACTIVE"; ob.active_material_index=0
print(bpy.ops.tk.split_by_material()); report("ACTIVE COPY on_copy:")
# export
d=tempfile.mkdtemp(); s.export_dir=d
for fmt in ("FBX","OBJ","GLB"):
    s.export_format=fmt; [o.select_set(o.name.startswith("Box_")) for o in bpy.data.objects]
    r=bpy.ops.tk.split_export(); print(fmt, r)
print(sorted(os.listdir(d)))
# highlight of the chosen material in the viewport (geometry part; drawing needs a GPU)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
ob, ms = setup(); s.mode = "ACTIVE"; ob.active_material_index = 1
sets = S.highlight_sets(ob, s); print("highlight sets", sets)
assert len(sets) == 1 and sets[0][1] == {1}
nface = sum(1 for p in ob.data.polygons if p.material_index == 1)
tris, seg = S.material_region(ob, bpy.context.evaluated_depsgraph_get(), {1})
print("highlight", nface, "faces ->", len(tris), "tri verts,", len(seg), "outline verts")
assert len(tris) == nface * 2 * 3 and len(seg) > 0
s.mode = "PICKED"; ms[2].tk_split_pick = True
sets = S.highlight_sets(ob, s); print("picked sets", sets)
assert {frozenset(x[1]) for x in sets} == {frozenset({1}), frozenset({2})}
