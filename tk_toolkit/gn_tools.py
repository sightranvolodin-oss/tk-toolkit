"""Artist-friendly Geometry Nodes tools: ready-made node groups with plain sliders."""

import math

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty
from mathutils import Vector

from . import common as C

S = C.enabled_socket


def _iface(ng, name, in_out, stype, default=None, minv=None, maxv=None, subtype=None):
    item = ng.interface.new_socket(name=name, in_out=in_out, socket_type=stype)
    if default is not None and hasattr(item, "default_value"):
        item.default_value = default
    if minv is not None and hasattr(item, "min_value"):
        item.min_value = minv
    if maxv is not None and hasattr(item, "max_value"):
        item.max_value = maxv
    if subtype and hasattr(item, "subtype"):
        try:
            item.subtype = subtype
        except TypeError:
            pass
    return item


def _new(ng, idname, x, y, **props):
    n = ng.nodes.new(idname)
    n.location = (x, y)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _random(ng, x, y, dtype, seed):
    n = _new(ng, "FunctionNodeRandomValue", x, y, data_type=dtype)
    S(n.inputs, "Seed").default_value = seed
    return n


def _rotation_block(ng, x, y, align_socket, aligned_rotation, z_angle_socket=None, z_angle_value=None):
    """Switch(aligned / identity) then rotate locally around Z."""
    sw = _new(ng, "GeometryNodeSwitch", x, y, input_type="ROTATION")
    ng.links.new(align_socket, S(sw.inputs, "Switch"))
    ng.links.new(aligned_rotation, S(sw.inputs, "True"))
    comb = _new(ng, "ShaderNodeCombineXYZ", x, y - 200)
    if z_angle_socket is not None:
        ng.links.new(z_angle_socket, comb.inputs["Z"])
    e2r = _new(ng, "FunctionNodeEulerToRotation", x + 180, y - 200)
    ng.links.new(comb.outputs[0], e2r.inputs[0])
    rot = _new(ng, "FunctionNodeRotateRotation", x + 360, y)
    if hasattr(rot, "rotation_space"):
        rot.rotation_space = "LOCAL"
    ng.links.new(S(sw.outputs, "Output"), S(rot.inputs, "Rotation"))
    ng.links.new(e2r.outputs[0], S(rot.inputs, "Rotate By"))
    return rot.outputs[0]


def build_scatter_group():
    name = "TK Scatter"
    ng = bpy.data.node_groups.get(name)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    _iface(ng, "Geometry", "INPUT", "NodeSocketGeometry")
    _iface(ng, "Объекты", "INPUT", "NodeSocketCollection")
    _iface(ng, "Плотность", "INPUT", "NodeSocketFloat", 10.0, 0.0, 100000.0)
    _iface(ng, "Seed", "INPUT", "NodeSocketInt", 0)
    _iface(ng, "Масштаб от", "INPUT", "NodeSocketFloat", 0.8, 0.0)
    _iface(ng, "Масштаб до", "INPUT", "NodeSocketFloat", 1.2, 0.0)
    _iface(ng, "Случайный поворот", "INPUT", "NodeSocketFloat", math.pi, 0.0, 2 * math.pi, "ANGLE")
    _iface(ng, "По нормали", "INPUT", "NodeSocketBool", True)
    _iface(ng, "Оставить поверхность", "INPUT", "NodeSocketBool", True)
    _iface(ng, "Geometry", "OUTPUT", "NodeSocketGeometry")

    gi = _new(ng, "NodeGroupInput", -900, 0)
    go = _new(ng, "NodeGroupOutput", 900, 0)
    dist = _new(ng, "GeometryNodeDistributePointsOnFaces", -600, 200)
    ng.links.new(gi.outputs["Geometry"], dist.inputs["Mesh"])
    ng.links.new(gi.outputs["Плотность"], S(dist.inputs, "Density"))
    ng.links.new(gi.outputs["Seed"], S(dist.inputs, "Seed"))

    coll = _new(ng, "GeometryNodeCollectionInfo", -600, -150)
    coll.transform_space = "ORIGINAL"
    ng.links.new(gi.outputs["Объекты"], coll.inputs["Collection"])
    coll.inputs["Separate Children"].default_value = True
    coll.inputs["Reset Children"].default_value = True

    ri = _random(ng, -350, -350, "INT", 3)
    S(ri.inputs, "Max").default_value = 10000
    rs = _random(ng, -350, -550, "FLOAT", 5)
    ng.links.new(gi.outputs["Масштаб от"], S(rs.inputs, "Min"))
    ng.links.new(gi.outputs["Масштаб до"], S(rs.inputs, "Max"))
    rr = _random(ng, -600, -500, "FLOAT", 7)
    ng.links.new(gi.outputs["Случайный поворот"], S(rr.inputs, "Max"))

    rot = _rotation_block(ng, -350, 50, gi.outputs["По нормали"], S(dist.outputs, "Rotation"),
                          S(rr.outputs, "Value"))

    inst = _new(ng, "GeometryNodeInstanceOnPoints", 200, 100)
    ng.links.new(dist.outputs["Points"], inst.inputs["Points"])
    ng.links.new(coll.outputs[0], inst.inputs["Instance"])
    inst.inputs["Pick Instance"].default_value = True
    ng.links.new(S(ri.outputs, "Value"), inst.inputs["Instance Index"])
    ng.links.new(rot, inst.inputs["Rotation"])
    ng.links.new(S(rs.outputs, "Value"), inst.inputs["Scale"])

    keep = _new(ng, "GeometryNodeSwitch", 200, 350, input_type="GEOMETRY")
    ng.links.new(gi.outputs["Оставить поверхность"], S(keep.inputs, "Switch"))
    ng.links.new(gi.outputs["Geometry"], S(keep.inputs, "True"))
    join = _new(ng, "GeometryNodeJoinGeometry", 600, 0)
    ng.links.new(S(keep.outputs, "Output"), join.inputs[0])
    ng.links.new(inst.outputs[0], join.inputs[0])
    ng.links.new(join.outputs[0], go.inputs[0])
    return ng


def build_along_curve_group(mesh_input=False):
    name = "TK Along Curve" + (" (mesh)" if mesh_input else "")
    ng = bpy.data.node_groups.get(name)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    _iface(ng, "Geometry", "INPUT", "NodeSocketGeometry")
    _iface(ng, "Объект", "INPUT", "NodeSocketObject")
    _iface(ng, "Шаг", "INPUT", "NodeSocketFloat", 0.5, 0.001, 1000.0, "DISTANCE")
    _iface(ng, "Поворот", "INPUT", "NodeSocketFloat", 0.0, -2 * math.pi, 2 * math.pi, "ANGLE")
    _iface(ng, "Масштаб", "INPUT", "NodeSocketFloat", 1.0, 0.0)
    _iface(ng, "По направлению кривой", "INPUT", "NodeSocketBool", True)
    _iface(ng, "Оставить кривую", "INPUT", "NodeSocketBool", False)
    _iface(ng, "Geometry", "OUTPUT", "NodeSocketGeometry")

    gi = _new(ng, "NodeGroupInput", -1000, 0)
    go = _new(ng, "NodeGroupOutput", 900, 0)
    src = gi.outputs["Geometry"]
    if mesh_input:
        m2c = _new(ng, "GeometryNodeMeshToCurve", -800, 200)
        ng.links.new(src, m2c.inputs[0])
        src = m2c.outputs[0]
    c2p = _new(ng, "GeometryNodeCurveToPoints", -600, 200, mode="LENGTH")
    ng.links.new(src, c2p.inputs["Curve"])
    ng.links.new(gi.outputs["Шаг"], S(c2p.inputs, "Length"))

    oi = _new(ng, "GeometryNodeObjectInfo", -600, -150)
    oi.transform_space = "ORIGINAL"
    ng.links.new(gi.outputs["Объект"], oi.inputs["Object"])
    oi.inputs["As Instance"].default_value = True

    rot = _rotation_block(ng, -350, 50, gi.outputs["По направлению кривой"],
                          S(c2p.outputs, "Rotation"), gi.outputs["Поворот"])
    inst = _new(ng, "GeometryNodeInstanceOnPoints", 200, 100)
    ng.links.new(c2p.outputs["Points"], inst.inputs["Points"])
    ng.links.new(S(oi.outputs, "Geometry"), inst.inputs["Instance"])
    ng.links.new(rot, inst.inputs["Rotation"])
    ng.links.new(gi.outputs["Масштаб"], inst.inputs["Scale"])

    keep = _new(ng, "GeometryNodeSwitch", 200, 350, input_type="GEOMETRY")
    ng.links.new(gi.outputs["Оставить кривую"], S(keep.inputs, "Switch"))
    ng.links.new(gi.outputs["Geometry"], S(keep.inputs, "True"))
    join = _new(ng, "GeometryNodeJoinGeometry", 600, 0)
    ng.links.new(S(keep.outputs, "Output"), join.inputs[0])
    ng.links.new(inst.outputs[0], join.inputs[0])
    ng.links.new(join.outputs[0], go.inputs[0])
    return ng


def set_input(mod, name, value):
    """Set a Geometry Nodes modifier input by its interface name (Blender 4.x and 5.x)."""
    for item in mod.node_group.interface.items_tree:
        if getattr(item, "in_out", "") == "INPUT" and item.name == name:
            ident = item.identifier
            props = getattr(mod, "properties", None)
            inputs = getattr(props, "inputs", None) if props is not None else None
            if inputs is not None and hasattr(inputs, ident):
                getattr(inputs, ident).value = value      # Blender 5.x
            else:
                mod[ident] = value                         # Blender 4.x
            return True
    return False


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class TK_OT_gn_scatter(bpy.types.Operator):
    bl_idname = "tk.gn_scatter"
    bl_label = "Рассыпать по поверхности"
    bl_description = ("Выделите объекты-источники (камни, траву…), затем поверхность — активной. "
                      "Получите модификатор с ползунками плотности, масштаба и поворота")
    bl_options = {"REGISTER", "UNDO"}

    density: FloatProperty(name="Плотность", default=10.0, min=0.0)
    scale_min: FloatProperty(name="Масштаб от", default=0.8, min=0.0)
    scale_max: FloatProperty(name="Масштаб до", default=1.2, min=0.0)
    hide_sources: BoolProperty(name="Скрыть исходники", default=True)

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and context.mode == "OBJECT"

    def execute(self, context):
        surf = context.active_object
        srcs = [o for o in context.selected_objects if o is not surf]
        if not srcs:
            self.report({"WARNING"}, "Выделите объекты для рассыпки, затем поверхность (активной)")
            return {"CANCELLED"}
        coll = bpy.data.collections.new(f"TK_scatter_{surf.name}")
        for o in srcs:
            coll.objects.link(o)
            if self.hide_sources:
                o.hide_set(True)
                o.hide_render = True
        mod = surf.modifiers.new("TK Рассыпка", "NODES")
        mod.node_group = build_scatter_group()
        set_input(mod, "Объекты", coll)
        set_input(mod, "Плотность", self.density)
        set_input(mod, "Масштаб от", self.scale_min)
        set_input(mod, "Масштаб до", self.scale_max)
        surf.update_tag()
        self.report({"INFO"}, "Настраивайте в модификаторе «TK Рассыпка»")
        return {"FINISHED"}


class TK_OT_gn_along_curve(bpy.types.Operator):
    bl_idname = "tk.gn_along_curve"
    bl_label = "Расставить вдоль кривой"
    bl_description = ("Выделите объект (столбик, звено цепи…), затем кривую или меш-линию — "
                      "активной. Объект повторится вдоль неё с заданным шагом")
    bl_options = {"REGISTER", "UNDO"}

    spacing: FloatProperty(name="Шаг", default=0.5, min=0.001, subtype="DISTANCE")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type in {"CURVE", "MESH"} and context.mode == "OBJECT"

    def execute(self, context):
        path = context.active_object
        items = [o for o in context.selected_objects if o is not path]
        if not items:
            self.report({"WARNING"}, "Выделите объект для повтора, затем кривую (активной)")
            return {"CANCELLED"}
        mod = path.modifiers.new("TK Вдоль кривой", "NODES")
        mod.node_group = build_along_curve_group(mesh_input=path.type == "MESH")
        set_input(mod, "Объект", items[0])
        set_input(mod, "Шаг", self.spacing)
        path.update_tag()
        return {"FINISHED"}


def cable_points(a, b, sag, n):
    pts = []
    for i in range(n + 1):
        t = i / n
        p = a.lerp(b, t)
        p.z -= sag * 4.0 * t * (1.0 - t)
        pts.append(p)
    return pts


class TK_OT_cable(bpy.types.Operator):
    bl_idname = "tk.cable"
    bl_label = "Провода между объектами"
    bl_description = "Провисающие провода между выделенными объектами (цепочкой от активного)"
    bl_options = {"REGISTER", "UNDO"}

    sag: FloatProperty(name="Провисание", default=0.3, min=0.0, subtype="DISTANCE")
    thickness: FloatProperty(name="Толщина", default=0.01, min=0.0, subtype="DISTANCE")
    segments: IntProperty(name="Сегментов", default=24, min=2, max=256)

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and len(context.selected_objects) >= 2

    def execute(self, context):
        objs = list(context.selected_objects)
        act = context.active_object if context.active_object in objs else objs[0]
        chain = [act]
        rest = [o for o in objs if o is not act]
        while rest:
            last = chain[-1].matrix_world.translation
            nxt = min(rest, key=lambda o: (o.matrix_world.translation - last).length)
            chain.append(nxt)
            rest.remove(nxt)
        cu = bpy.data.curves.new("TK_cable", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = self.thickness
        cu.bevel_resolution = 3
        cu.use_fill_caps = True
        for a, b in zip(chain, chain[1:]):
            pts = cable_points(a.matrix_world.translation.copy(), b.matrix_world.translation.copy(),
                               self.sag, self.segments)
            sp = cu.splines.new("POLY")
            sp.points.add(len(pts) - 1)
            for p, sp_pt in zip(pts, sp.points):
                sp_pt.co = (p.x, p.y, p.z, 1.0)
            sp.use_smooth = True
        ob = bpy.data.objects.new("TK_cable", cu)
        context.collection.objects.link(ob)
        return {"FINISHED"}


class TK_PT_gn_tools(bpy.types.Panel):
    bl_label = "Готовые инструменты (GN)"
    bl_idname = "TK_PT_gn_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 6

    def draw(self, context):
        col = self.layout.column(align=True)
        col.scale_y = 1.2
        col.operator("tk.gn_scatter", icon="PARTICLES")
        col.operator("tk.gn_along_curve", icon="CURVE_PATH")
        col.operator("tk.cable", icon="OUTLINER_DATA_CURVE")
        self.layout.label(text="Настройки — в панели модификатора", icon="INFO")


classes = (TK_OT_gn_scatter, TK_OT_gn_along_curve, TK_OT_cable, TK_PT_gn_tools)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
