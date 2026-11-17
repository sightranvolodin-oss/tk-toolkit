import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, bpy, bmesh
sys.path.insert(0, ROOT)
from mathutils import Vector
from tk_toolkit import retopo as R
R.register()
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=32, ring_count=16)
ref=bpy.context.object; ref.name="high"; ref.location=(1,2,0)
bpy.ops.tk.retopo_setup(mirror_x=False)
obj=bpy.context.object
print("mode",bpy.context.mode, obj.name, [m.type for m in obj.modifiers], ref.hide_select)
proj=R.Projector(ref,obj,bpy.context.evaluated_depsgraph_get())
bm=bmesh.from_edit_mesh(obj.data)
# bowtie click order: a,c,b,d
pts=[Vector((0.3,0.3,0.9)),Vector((-0.3,-0.3,0.9)),Vector((-0.3,0.3,0.9)),Vector((0.3,-0.3,0.9))]
vs=[bm.verts.new(proj.nearest(p)[0]) for p in pts]
n=proj.nearest(Vector((0,0,1)))[1]
f=R.make_face(bm,vs,n)
print("face ok", f is not None, "normal·ref", round(f.normal.dot(n),3), "len", [round(v.co.length,3) for v in f.verts])
# simple test: is quad non-self-intersecting -> area close to 0.36ish
print("area", round(f.calc_area(),3))
e=[e for e in f.edges][0]
f2=R.extend_edge(bm,e,(e.verts[0].co+e.verts[1].co)/2*0.0+ ( (e.verts[0].co+e.verts[1].co)/2 - f.calc_center_median())*2, project=lambda p: proj.nearest(p)[0], ref_normal=n)
print("extend", f2 is not None, len(bm.faces), "consistent", f2.normal.dot(f.normal)>0)
R.relax_verts(bm, list(bm.verts), 3, 0.5, True, lambda p: proj.nearest(p)[0])
print("on surface", all(abs(v.co.length-1)<0.01 for v in bm.verts))
bmesh.update_edit_mesh(obj.data)
bpy.ops.tk.retopo_relax(iterations=2)
bpy.ops.tk.retopo_finish()
print("finished", bpy.context.mode, [m.type for m in obj.modifiers], ref.hide_select, len(obj.data.polygons))
