import sys, os, runpy, traceback
try: script=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(script, run_name="__main__")
except SystemExit: pass
except Exception: traceback.print_exc()
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
