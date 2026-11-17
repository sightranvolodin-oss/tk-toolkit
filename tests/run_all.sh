#!/usr/bin/env bash
# Runs every test with Blender-as-a-Python-module (pip install "bpy>=4.5,<4.6" numpy).
# A test passes only if it reaches its end (run.py prints __TEST_OK__) without a traceback.
set -u
PY="${PYTHON:-python}"
cd "$(dirname "$0")"
fail=0
for t in t_ui t_split t_maps t_atlas_check t_all t_bake t_layers t_modes t_retopo; do
  echo "=== $t"
  out=$(timeout 900 "$PY" run.py "$t.py" 2>&1 | grep -v "^Fra:")
  rc=${PIPESTATUS[0]}
  echo "$out" | grep -v -i "unfreed\|leak detector\|^$" | tail -25
  if ! echo "$out" | grep -q "__TEST_OK__" || echo "$out" | grep -q -E "Traceback|Error: Python"; then
    echo "!!! $t FAILED (exit code $rc)"
    fail=1
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      msg=$(echo "$out" | grep -v -i "unfreed\|leak detector\|^$\|^Info:\|Baking map" | tail -30 \
            | sed -e 's/%/%25/g' -e 's/\r//g' | awk '{printf "%s%%0A", $0}')
      echo "::error title=$t failed::${msg}"
    fi
  fi
done
exit $fail
