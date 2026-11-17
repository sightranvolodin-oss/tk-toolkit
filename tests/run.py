"""Run one test script inside Blender-as-a-module and exit without Blender's slow teardown.

Prints __TEST_OK__ only when the script finished without an exception, so a crash or a
silent failure is never mistaken for success.
"""
import os
import runpy
import sys
import traceback

ok = False
try:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
    ok = True
except SystemExit:
    ok = True
except Exception as ex:
    traceback.print_exc()
    tb = traceback.extract_tb(ex.__traceback__)
    where = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in reversed(tb[-4:]))
    print(f"__TEST_FAIL__ {type(ex).__name__}: {str(ex).splitlines()[-1] if str(ex) else ''} @ {where}")
if ok:
    print("__TEST_OK__")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0 if ok else 1)
