#!/usr/bin/env bash
# Runs every test with Blender-as-a-Python-module (pip install bpy==4.5.* numpy).
# A test fails when its output contains a Python traceback.
set -u
PY="${PYTHON:-python}"
cd "$(dirname "$0")"
fail=0
for t in t_ui t_split t_maps t_atlas_check t_all t_bake t_layers t_modes t_retopo; do
  echo "=== $t"
  out=$(timeout 600 "$PY" run.py "$t.py" 2>&1)
  echo "$out" | grep -v -i "unfreed\|leak detector\|^$" | tail -25
  if echo "$out" | grep -q -E "Traceback|Error: Python"; then
    echo "!!! $t FAILED"; fail=1
  fi
done
exit $fail
