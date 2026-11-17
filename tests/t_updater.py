import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, bpy
sys.path.insert(0, ROOT)
from tk_toolkit import updater as U
print("current", U.current_version(), "parse", U.parse_version("v0.10.2") > U.parse_version("0.9.9"))
print("newer?", U.check("sightranvolodin-oss/tk-toolkit"), U._state)
