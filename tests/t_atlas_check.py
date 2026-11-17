import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, bpy, numpy as np, tempfile
sys.path.insert(0, ROOT)
from tk_toolkit import maps as M
M.register()
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc=bpy.context.scene; p=sc.tk_maps; p.output_dir=tempfile.mkdtemp()
N=64; yy,xx=np.mgrid[0:N,0:N]
def img(name, f):
    a=np.ones((N,N,4),np.float32); a[...,:3]=f; im=bpy.data.images.new(name,N,N); im.pixels.foreach_set(a.ravel()); im.update(); return im
def mat(name, image):
    m=bpy.data.materials.new(name)
    try: m.use_nodes=True
    except Exception: pass
    nt=m.node_tree; b=M.analyse(m)["shader"]; t=nt.nodes.new("ShaderNodeTexImage"); t.image=image
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"]); return m
A=img("A", np.stack([xx/N, yy/N, np.full((N,N),0.2)],-1))
B=img("B", np.stack([np.full((N,N),0.9), (xx+yy)/(2*N), yy/N],-1))
for tiling in (1.0, 2.0):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4); ob=bpy.context.object
    ob.data.materials.append(mat("MA",A)); ob.data.materials.append(mat("MB",B))
    for i,poly in enumerate(ob.data.polygons): poly.material_index = 0 if poly.center.x<0 else 1
    uv=ob.data.uv_layers[0]; buf=np.empty(len(ob.data.loops)*2,np.float32); uv.data.foreach_get("uv",buf); uv.data.foreach_set("uv",buf*tiling)
    def bake_ref(o, uvname):
        im=M.new_image("ref_"+o.name,128,128,"DIFFUSE"); M.bake_channel(bpy.context,o,"DIFFUSE",im,margin=0,uv_layer=uvname); a=M.image_array(im).copy(); bpy.data.images.remove(im); return a
    ref=bake_ref(ob,"UVMap")
    for mode in ("GRID","REPACK"):
        if tiling>1 and mode=="GRID": continue
        p.atlas_mode=mode; p.atlas_size="1024"; p.ch_opacity=False
        for o in bpy.context.selected_objects: o.select_set(False)
        ob.hide_set(False); ob.select_set(True); bpy.context.view_layer.objects.active=ob
        bpy.ops.tk.maps_atlas(); new=bpy.data.objects[ob.name+"_Atlas"]
        # sample atlas material through ORIGINAL uv: material uses UVMap node UV_Atlas, so bake output in UVMap space
        got=bake_ref(new,"UVMap")
        cov=ref[...,3]>0
        err=np.abs(got[...,:3]-ref[...,:3]).max(-1)
        print("  ref std", round(float(ref[...,:3].std()),3), "got std", round(float(got[...,:3].std()),3), "mats", [m.name for m in new.data.materials], "tex", [n.image.name for n in new.data.materials[0].node_tree.nodes if n.bl_idname=="ShaderNodeTexImage"])
        print(f"tiling {tiling} {mode}: median err {np.median(err):.4f}, 99% {np.percentile(err,99):.4f}")
        bpy.data.objects.remove(new)
    bpy.data.objects.remove(ob)
