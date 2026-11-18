"""TK Toolkit — bake preview, retopology, texture maps & atlas, split by material,
texture layers, readable history,
scene profiler, precision tools and artist-friendly Geometry Nodes tools.

Sidebar (N) → tab «Toolkit».
"""

bl_info = {
    "name": "TK Toolkit",
    "author": "TK Toolkit",
    "version": (0, 5, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport → Sidebar (N) → Toolkit",
    "description": "Bake preview, retopology, texture maps & atlas, split by material, layers and more",
    "category": "3D View",
}

if "bpy" in locals():
    import importlib
    for _m in (common, bake_preview, retopo, layers, maps, cards, split, history, profiler, precision, gn_tools, updater):  # noqa: F821
        importlib.reload(_m)
else:
    from . import (common, bake_preview, retopo, layers, maps, cards, split, history, profiler,
                   precision, gn_tools, updater)

import bpy  # noqa: E402

modules = (bake_preview, retopo, layers, maps, cards, split, history, profiler, precision, gn_tools,
           updater)


def register():
    for m in modules:
        m.register()


def unregister():
    for m in reversed(modules):
        try:
            m.unregister()
        except Exception as ex:  # keep unregistering the rest
            print("TK Toolkit unregister:", m.__name__, ex)
