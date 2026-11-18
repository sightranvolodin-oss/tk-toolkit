#!/usr/bin/env bash
# Runs every test with Blender-as-a-Python-module (pip install "bpy>=4.5,<4.6" numpy).
# A test passes only if it reaches its end (run.py prints __TEST_OK__) without a traceback.
set -u
PY="${PYTHON:-python}"
cd "$(dirname "$0")"
fail=0
for t in t_ui t_cards t_railing t_batch t_facade_detect t_split t_maps t_atlas_check t_all t_bake t_layers t_modes t_retopo; do
  echo "=== $t"
  out=$(timeout 900 "$PY" run.py "$t.py" 2>&1 | grep -v "^Fra:")
  rc=${PIPESTATUS[0]}
  echo "$out" | grep -v -i "unfreed\|leak detector\|^$" | tail -25
  if ! echo "$out" | grep -q "__TEST_OK__" || echo "$out" | grep -q -E "Traceback|Error: Python"; then
    echo "!!! $t FAILED (exit code $rc)"
    fail=1
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      line=$(echo "$out" | grep -m1 "__TEST_FAIL__" | sed 's/%/%25/g')
      if [ -z "$line" ]; then
        line=$(echo "$out" | grep -A12 -m1 "Fatal Python error" | grep -E "Fatal|File " \
               | sed -e 's/^ *//' -e 's/%/%25/g' | tr '\n' ' ' | cut -c1-900)
      fi
      if [ -z "$line" ]; then
        line="crash: $(echo "$out" | grep -v -i "unfreed\|leak detector\|^$\|Baking map\|^Info" | tail -3 | tr '\n' ' ' | cut -c1-600)"
      fi
      echo "::error title=$t failed::${line:-no __TEST_OK__ (crash?) exit $rc}"
    fi
  fi
done
exit $fail
