import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, bpy
sys.path.insert(0, ROOT)
from tk_toolkit import bake_preview as BP
BP.register()
bpy.ops.wm.read_factory_settings(use_empty=True) if False else None
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc=bpy.context.scene
bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0); low=bpy.context.object; low.name="low"
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=1.03); high=bpy.context.object; high.name="high"
tex=bpy.data.textures.new("n","CLOUDS"); tex.noise_scale=0.3
dm=high.modifiers.new("d","DISPLACE"); dm.texture=tex; dm.strength=0.05; dm.mid_level=0.5
low.select_set(True); high.select_set(True); bpy.context.view_layer.objects.active=low
s=sc.tk_bake
dg=bpy.context.evaluated_depsgraph_get()
def run(label):
    res,err=BP.compute(sc,bpy.context.view_layer,bpy.context.evaluated_depsgraph_get(),s)
    if err: print(label,"ERR",err); return
    st=res["stats"]; print(label, {k:st[k] for k in ("n","ok","miss","back","warn","flat_faces")}, "hints",st["hints"], f"{st['time']*1000:.0f}ms")
sc.render.bake.cage_extrusion=0; sc.render.bake.max_ray_distance=0
run("e=0,d=0 flat")
e,d,u=BP.autofit(sc,bpy.context.view_layer,dg)
print("autofit",round(e,4),round(d,4),u)
sc.render.bake.cage_extrusion=e; sc.render.bake.max_ray_distance=d
run("autofit flat")
for p in low.data.polygons: p.use_smooth=True
low.data.update()
BP._state["geo_ver"]["low"]=1
run("autofit smooth")
sc.render.bake.max_ray_distance=e*0.5
run("d<e")
# cage
bpy.ops.object.select_all(action='DESELECT')
cage=low.copy(); cage.data=low.data.copy(); cage.name="cage"; sc.collection.objects.link(cage)
sm=cage.modifiers.new("disp","DISPLACE"); sm.strength=0.12; sm.mid_level=0
sc.render.bake.use_cage=True; sc.render.bake.cage_object=cage; sc.render.bake.max_ray_distance=0
low.select_set(True); high.select_set(True); bpy.context.view_layer.objects.active=low
run("cage")
# probe headless math
print("poly index test:", BP.C.object_bvh(low, bpy.context.evaluated_depsgraph_get())[0].ray_cast((0,0,5),(0,0,-1)))
