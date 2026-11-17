import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, bpy, numpy as np, uuid
sys.path.insert(0, ROOT)
from tk_toolkit import layers as LY
LY.register()
bpy.ops.mesh.primitive_plane_add(size=2); obj=bpy.context.object
sc=bpy.context.scene; sc.render.engine="CYCLES"; sc.cycles.samples=1; sc.render.bake.margin=0
N=16; rng=np.random.default_rng(5)
def mkimg(name, alpha=True):
    im=bpy.data.images.new(name,N,N,alpha=True)
    px=rng.random((N,N,4)).astype(np.float32)
    if not alpha: px[...,3]=1
    im.pixels.foreach_set(px.ravel()); im.update(); return im
for m in ["MIX","MULTIPLY","SCREEN","OVERLAY","ADD","SUBTRACT","DARKEN","LIGHTEN"]:
    mat=bpy.data.materials.new(m); LY.ensure_tree(mat); obj.data.materials.clear(); obj.data.materials.append(mat)
    bsdf=LY.find_principled(mat.node_tree); bsdf.inputs["Base Color"].default_value=(0.3,0.5,0.7,1)
    LY._building["on"]=True
    l=mat.tk_layers.add(); l.uid=uuid.uuid4().hex[:8]; l.blend=m; l.image=mkimg(m); l.opacity=0.7
    LY._building["on"]=False; LY.rebuild(mat); nt=mat.node_tree
    ref=LY.composite(mat,"BASE_COLOR",N,N)
    src=bsdf.inputs["Base Color"].links[0].from_socket
    em=nt.nodes.new("ShaderNodeEmission"); out=[n for n in nt.nodes if n.bl_idname=="ShaderNodeOutputMaterial"][0]
    nt.links.new(src, em.inputs[0]); nt.links.new(em.outputs[0], out.inputs[0])
    tgt=bpy.data.images.new("bake"+m,N,N,float_buffer=True); tn=nt.nodes.new("ShaderNodeTexImage"); tn.image=tgt; nt.nodes.active=tn
    bpy.ops.object.bake(type="EMIT")
    b=np.empty(N*N*4,np.float32); tgt.pixels.foreach_get(b); b=b.reshape(N,N,4)[...,:3]
    print(m, "max", round(float(np.abs(b-ref).max()),5))
