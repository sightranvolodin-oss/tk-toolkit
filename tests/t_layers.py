import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, bpy, numpy as np, uuid
sys.path.insert(0, ROOT)
from tk_toolkit import layers as LY
LY.register()
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.mesh.primitive_plane_add(size=2); obj=bpy.context.object
mat=bpy.data.materials.new("M"); LY.ensure_tree(mat); obj.data.materials.append(mat)
bsdf=LY.find_principled(mat.node_tree); bsdf.inputs["Base Color"].default_value=(0.3,0.5,0.7,1)
rng=np.random.default_rng(3); N=32
def mkimg(name, alpha=True):
    im=bpy.data.images.new(name,N,N,alpha=True)
    px=rng.random((N,N,4)).astype(np.float32)
    if not alpha: px[...,3]=1
    im.pixels.foreach_set(px.ravel()); im.update(); return im
modes=["MIX","MULTIPLY","SCREEN","OVERLAY","ADD","SUBTRACT","DARKEN","LIGHTEN"]
LY._building["on"]=True
for i,m in enumerate(modes):
    l=mat.tk_layers.add(); l.uid=uuid.uuid4().hex[:8]; l.name=m; l.blend=m; l.image=mkimg(m); l.opacity=0.4+0.07*i
    if m=="SCREEN": l.mask=mkimg("mask",alpha=False)
LY._building["on"]=False
LY.rebuild(mat)
nt=mat.node_tree
print("nodes", sum(1 for n in nt.nodes if n.get(LY.FLAG)), "base linked", bsdf.inputs["Base Color"].is_linked)
# numpy flatten
ref=LY.composite(mat,"BASE_COLOR",N,N)
# cycles bake of base color via emission
src=bsdf.inputs["Base Color"].links[0].from_socket
em=nt.nodes.new("ShaderNodeEmission"); out=[n for n in nt.nodes if n.bl_idname=="ShaderNodeOutputMaterial"][0]
nt.links.new(src, em.inputs[0]); nt.links.new(em.outputs[0], out.inputs[0])
tgt=bpy.data.images.new("bake",N,N,float_buffer=True)
tn=nt.nodes.new("ShaderNodeTexImage"); tn.image=tgt; nt.nodes.active=tn
sc=bpy.context.scene; sc.render.engine="CYCLES"; sc.cycles.samples=1; sc.cycles.device="CPU"
sc.render.bake.margin=0
bpy.ops.object.bake(type="EMIT")
b=np.empty(N*N*4,np.float32); tgt.pixels.foreach_get(b); b=b.reshape(N,N,4)[...,:3]
d=np.abs(b-ref)
print("max diff vs Cycles", float(d.max()), "mean", float(d.mean()))
# flatten operator + replace
with bpy.context.temp_override(active_object=obj, object=obj):
    bpy.ops.tk.layers_flatten(channel="BASE_COLOR", replace=True)
print("after replace layers:", [l.name for l in mat.tk_layers], bpy.data.images.get("M_base_color_flat").packed_file is not None)
