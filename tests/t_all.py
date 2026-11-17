import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, bpy, bmesh, math
sys.path.insert(0, ROOT)
import tk_toolkit as TK
TK.register()
print("registered OK")
from tk_toolkit import history as H, profiler as P, precision as PR, gn_tools as G
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
# ---- history
bpy.ops.mesh.primitive_cube_add()
bpy.ops.transform.translate(value=(0,0,0.25), constraint_axis=(False,False,True))
bpy.ops.transform.rotate(value=math.radians(30), orient_axis='Z')
wm=bpy.context.window_manager
print("wm.operators:", [ (o.bl_idname, 'UNDO' in o.bl_options) for o in wm.operators][-3:])
for op in list(wm.operators)[-3:]: print("  describe:", H.describe(op))
# ---- profiler
cube=bpy.context.object
m=cube.modifiers.new("S","SUBSURF"); m.levels=6
for i in range(3):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=64, location=(i*3,5,0))
img=bpy.data.images.new("big",8192,8192)
items,summ=P.analyze(bpy.context)
for d in items[:6]: print("  prof:", d["kind"], d["target"], "|", d["info"], "| fix:", d["fix"])
print("  summary", {k:(round(v,1) if isinstance(v,float) else v) for k,v in summ.items()})
bpy.ops.tk.profile_scene()
sc=bpy.context.scene
idx=[i for i,it in enumerate(sc.tk_prof_items) if it.fix=="SUBSURF_VIEWPORT"][0]
bpy.ops.tk.profile_fix(index=idx); print("  subsurf after fix:", m.levels)
idx=[i for i,it in enumerate(sc.tk_prof_items) if it.fix=="LINK_MESHES"]
if idx: bpy.ops.tk.profile_fix(index=idx[0]); print("  meshes users:", [o.data.name for o in bpy.data.objects if o.name.startswith("Sphere")])
# ---- precision
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_plane_add(size=2); pl=bpy.context.object; pl.scale=(2,1,1)
bpy.ops.object.mode_set(mode='EDIT'); bm=bmesh.from_edit_mesh(pl.data)
for v in bm.verts: v.select=False
for e in bm.edges: e.select=False
bm.edges.ensure_lookup_table(); bm.verts.ensure_lookup_table(); e=bm.edges[0]; e.select=True
PR.set_edge_length(bm, pl.matrix_world, [e], 1.5)
mw=pl.matrix_world; print("  edge world len:", round((mw@e.verts[0].co - mw@e.verts[1].co).length,6))
vs=list(bm.verts)[:3]
PR.set_angle(mw, vs[0], vs[1], vs[2], math.radians(60))
print("  angle:", round(math.degrees(PR.angle_at(*(mw@v.co for v in vs))),4))
print("  quantize:", PR.quantize(__import__('mathutils').Vector((0.123,0.456,-0.789)),0.05))
bpy.ops.object.mode_set(mode='OBJECT')
# ---- gn
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_grid_add(size=4); ground=bpy.context.object
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.1, location=(10,0,0)); rock=bpy.context.object
rock.select_set(True); ground.select_set(True); bpy.context.view_layer.objects.active=ground
bpy.ops.tk.gn_scatter(density=20)
dg=bpy.context.evaluated_depsgraph_get()
inst=sum(1 for i in dg.object_instances if i.is_instance and i.parent and i.parent.original==ground)
print("  scatter instances:", inst)
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.curve.primitive_bezier_curve_add(); cu=bpy.context.object
bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0,10,0)); post=bpy.context.object
post.select_set(True); cu.select_set(True); bpy.context.view_layer.objects.active=cu
bpy.ops.tk.gn_along_curve(spacing=0.25)
dg=bpy.context.evaluated_depsgraph_get()
print("  along-curve instances:", sum(1 for i in dg.object_instances if i.is_instance and i.parent and i.parent.original==cu))
bpy.ops.object.select_all(action='DESELECT')
for o in (rock,post): o.hide_set(False); o.select_set(True)
bpy.context.view_layer.objects.active=rock
bpy.ops.tk.cable(sag=0.5)
c=[o for o in bpy.data.objects if o.name.startswith("TK_cable")][0]
print("  cable splines:", len(c.data.splines), "points", len(c.data.splines[0].points))
TK.unregister(); print("unregistered OK")
# history with fake operators
import types
TK.register()
def fake(idname,label,**props): return types.SimpleNamespace(bl_idname=idname,bl_label=label,bl_options={'REGISTER','UNDO'},properties=types.SimpleNamespace(**props))
from mathutils import Vector
ops=[fake("TRANSFORM_OT_translate","Move",value=Vector((0,0,0.25)),constraint_axis=(False,False,True)),
     fake("TRANSFORM_OT_rotate","Rotate",value=math.radians(30),orient_axis='Z'),
     fake("MESH_OT_extrude_region_move","Extrude",TRANSFORM_OT_translate=types.SimpleNamespace(value=Vector((0.1,0.2,0)),constraint_axis=(False,False,False))),
     fake("MESH_OT_bevel","Bevel",offset=0.02,segments=3),
     fake("MESH_OT_primitive_torus_add","Add Torus"),
     fake("SOME_OT_unknown","Some Thing")]
for o in ops: H._record(o)
print("history:", [H.entry_text(e) for e in H._log])
H._on_undo(); H._on_undo(); print("ptr after 2 undo:", H._ptr["i"], "-> next undo:", H.entry_text(H._log[H._ptr["i"]]))
H._record(fake("MESH_OT_delete","Delete",type="FACE")); print("after new op, len:", len(H._log), "last:", H.entry_text(H._log[-1]))
TK.unregister()
