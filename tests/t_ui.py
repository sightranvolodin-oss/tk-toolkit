import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, bpy, re, glob, types
sys.path.insert(0, ROOT)
import tk_toolkit as TK
TK.register()
icons={i.identifier for i in bpy.types.UILayout.bl_rna.functions['label'].parameters['icon'].enum_items}
bad=set()
for f in glob.glob(os.path.join(ROOT, "tk_toolkit", "*.py")):
    src=open(f).read()
    for m in re.findall(r'icon\s*=\s*"([A-Z0-9_]+)"', src)+re.findall(r'"([A-Z0-9_]+)"\s+if\s', src)+re.findall(r'else\s+"([A-Z0-9_]+)"', src):
        if m not in icons: bad.add((f.split('/')[-1],m))
    for m in re.findall(r'\("[A-Z_]+", "[^"]*", "[^"]*", "([A-Z0-9_]+)"', src):
        if m not in icons: bad.add((f.split('/')[-1],m))
print("bad icons:", bad or "none")
# stub layout that validates props/operators
errors=[]
class L:
    def __getattr__(self,n): return self._any
    def _any(self,*a,**k): return L()
    def prop(self,data,name,**k):
        if not hasattr(data,name) and name not in getattr(getattr(data,'bl_rna',None),'properties',{}): errors.append(("prop",type(data).__name__,name))
        return L()
    def operator(self,idname,**k):
        mod,_,op=idname.partition(".")
        try: getattr(getattr(bpy.ops,mod),op).get_rna_type()
        except Exception as e: errors.append(("op",idname,str(e)))
        return types.SimpleNamespace(direction=None,index=0)
    def template_list(self,*a,**k):
        data,prop=a[2],a[3]
        if not hasattr(data,prop): errors.append(("list",prop))
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.mesh.primitive_cube_add(); obj=bpy.context.object
mat=bpy.data.materials.new("M"); obj.data.materials.append(mat)
bpy.ops.tk.profile_scene()
from tk_toolkit import layers as LY
LY._building["on"]=True; l=mat.tk_layers.add(); l.uid="x"; LY._building["on"]=False
panels=[c for m in TK.modules for c in m.classes if issubclass(c,bpy.types.Panel)]
for P in panels:
    for mode in ("OBJECT","EDIT"):
        bpy.ops.object.mode_set(mode=mode)
        inst=types.SimpleNamespace(layout=L())
        try: P.draw(inst,bpy.context)
        except Exception as e: errors.append(("draw",P.__name__,mode,repr(e)))
    bpy.ops.object.mode_set(mode="OBJECT")
# UIList draw
for m in TK.modules:
    for c in m.classes:
        if issubclass(c,bpy.types.UIList):
            item = l if "layers" in c.__name__ else bpy.context.scene.tk_prof_items[0] if len(bpy.context.scene.tk_prof_items) else None
            if item is None: continue
            try: c.draw_item(types.SimpleNamespace(),bpy.context,L(),None,item,0,None,"",0)
            except Exception as e: errors.append(("uilist",c.__name__,repr(e)))
# bake panel with result
bpy.context.scene.tk_bake.samples=100
print("panels:", [p.__name__ for p in panels])
print("errors:", errors or "none")
