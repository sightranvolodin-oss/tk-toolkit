import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, bpy, numpy as np, tempfile
sys.path.insert(0, ROOT)
from tk_toolkit import maps as M
M.register()
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
out=tempfile.mkdtemp(); sc=bpy.context.scene; p=sc.tk_maps; p.output_dir=out
N=64; rng=np.random.default_rng(1)
# source leaf: sRGB colour + alpha disc
yy,xx=np.mgrid[0:N,0:N]; disc=((xx-32)**2+(yy-32)**2<20**2).astype(np.float32)
src=np.zeros((N,N,4),np.float32); src[...,0]=xx/N; src[...,1]=yy/N; src[...,2]=0.3; src[...,3]=disc
leaf=bpy.data.images.new("leaf",N,N,alpha=True); leaf.pixels.foreach_set(src.ravel()); leaf.update()
def mat_principled(name):
    m=bpy.data.materials.new(name)
    try: m.use_nodes=True
    except Exception: pass
    nt=m.node_tree; b=M.analyse(m)["shader"]
    t=nt.nodes.new("ShaderNodeTexImage"); t.image=leaf
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"]); nt.links.new(t.outputs["Alpha"], b.inputs["Alpha"])
    nz=nt.nodes.new("ShaderNodeTexNoise"); nt.links.new(nz.outputs[0], b.inputs["Roughness"])
    b.inputs["Metallic"].default_value=0.25
    return m
def mat_mixfoliage(name, transparent_first=True):
    m=bpy.data.materials.new(name)
    try: m.use_nodes=True
    except Exception: pass
    nt=m.node_tree
    for n in list(nt.nodes):
        if n.bl_idname!="ShaderNodeOutputMaterial": nt.nodes.remove(n)
    out_=[n for n in nt.nodes][0]
    mix=nt.nodes.new("ShaderNodeMixShader"); tr=nt.nodes.new("ShaderNodeBsdfTransparent"); d=nt.nodes.new("ShaderNodeBsdfDiffuse")
    t=nt.nodes.new("ShaderNodeTexImage"); t.image=leaf
    nt.links.new(t.outputs["Color"], d.inputs["Color"])
    fac_src=t.outputs["Alpha"]
    if not transparent_first:
        inv=nt.nodes.new("ShaderNodeMath"); inv.operation="SUBTRACT"; inv.inputs[0].default_value=1; nt.links.new(t.outputs["Alpha"],inv.inputs[1]); fac_src=inv.outputs[0]
    nt.links.new(fac_src, mix.inputs[0])
    a,b=(tr,d) if transparent_first else (d,tr)
    nt.links.new(a.outputs[0],mix.inputs[1]); nt.links.new(b.outputs[0],mix.inputs[2]); nt.links.new(mix.outputs[0],out_.inputs[0])
    return m
mp=mat_principled("Leaf_P"); mf=mat_mixfoliage("Leaf_Mix"); mf2=mat_mixfoliage("Leaf_Mix2",False)
bpy.ops.mesh.primitive_plane_add(); ob=bpy.context.object
for m in (mp,mf,mf2): ob.data.materials.append(m)
p.ch_roughness=True; p.ch_metallic=True; p.ch_normal=True; p.bleed=False; p.scope="ALL"; p.pack_rgba=True; p.pack_orm=True
r=bpy.ops.tk.maps_extract(); print("extract", r, sorted(os.listdir(out)))
def load(name):
    im=bpy.data.images.load(os.path.join(out,name)); a=M.image_array(im).copy(); return a
for base in ("Leaf_P","Leaf_Mix","Leaf_Mix2"):
    d=load(base+"_Diffuse.png"); o=load(base+"_Opacity.png")
    inside=disc>0.5
    print(base,"diffuse err(in leaf)",round(float(np.abs(d[...,:3][inside]-src[...,:3][inside]).max()),3),
          "opacity err",round(float(np.abs(o[...,0]-disc).max()),3))
r_=load("Leaf_P_Roughness.png")[...,0]; mt=load("Leaf_P_Metallic.png")[...,0]; nm=load("Leaf_P_Normal.png")
print("rough std (procedural)", round(float(r_.std()),3), "metal", round(float(mt.mean()),3), "normal mean", np.round(nm[...,:3].reshape(-1,3).mean(0),3))
rgba=load("Leaf_P_DiffuseOpacity.png"); print("packed alpha err", round(float(np.abs(rgba[...,3]-disc).max()),3))
# bleed check
p.bleed=True; p.bleed_px=40; p.pack_rgba=False; p.pack_orm=False; p.scope="ACTIVE"; ob.active_material_index=0
bpy.ops.tk.maps_extract(); d=load("Leaf_P_Diffuse.png"); print("bleed outside leaf red≈x/N:", round(float(d[2,2,0]),3), "(was", src[2,2,0],")")
# ---------- atlas
p.bleed=False; p.ch_normal=False; p.ch_ao=True; p.atlas_size="512"
me=ob.data
for i,poly in enumerate(me.polygons): poly.material_index=0
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.subdivide(number_cuts=1); bpy.ops.object.mode_set(mode="OBJECT")
for i,poly in enumerate(me.polygons): poly.material_index=i%3
for mode in ("GRID","REPACK"):
    p.atlas_mode=mode
    for o in bpy.context.selected_objects: o.select_set(False)
    ob.hide_set(False); ob.select_set(True); bpy.context.view_layer.objects.active=ob
    r=bpy.ops.tk.maps_atlas(); new=bpy.data.objects.get(ob.name+"_Atlas")
    print(mode, r, "obj", new and new.name, "mats", new and [m.name for m in new.data.materials], "uv", new and [u.name for u in new.data.uv_layers])
    # sanity: bake the new atlas material diffuse back via its own UV vs original
    a=load(f"{ob.name}_Atlas_Diffuse.png"); print("   atlas diffuse non-grey fraction", round(float((np.abs(a[...,:3]-0.5).sum(-1)>0.05).mean()),3))
    ao=os.path.exists(os.path.join(out,f"{ob.name}_Atlas_AO.png")); print("   AO file", ao)
    bpy.data.objects.remove(new)
