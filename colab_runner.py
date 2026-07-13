"""Colab runner: extract bundle, run colab.sh run, report exit status."""
import subprocess, sys, os, time

os.chdir("/content")

def dbg(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

dbg("========================================")
dbg("colab_runner.py started")
dbg(f"CWD: {os.getcwd()}")
dbg(f"Files in /content: {os.listdir('/content')}")
dbg(f"colab_bundle.tar.gz size: {os.path.getsize('/content/colab_bundle.tar.gz')} bytes")
dbg("========================================")

dbg("[1/2] Extracting bundle...")
extract_start = time.time()
subprocess.run(["tar", "xf", "colab_bundle.tar.gz"], check=True)
extract_elapsed = time.time() - extract_start
dbg(f"[1/2] Extract done in {extract_elapsed:.0f}s")
dbg(f"modeling_llama.py: {os.path.getsize('ONLY/patches/modeling_llama.py')} bytes")
dbg(f"colab.sh: {os.path.getsize('colab.sh')} bytes")

dbg("========================================")
dbg("[2/2] Running: bash colab.sh run")
dbg("========================================")

start = time.time()
result = subprocess.run(["bash", "colab.sh", "run"], capture_output=False)
elapsed = time.time() - start

dbg("========================================")
dbg(f"colab.sh run finished in {elapsed:.0f}s, exit code={result.returncode}")
dbg("========================================")
sys.exit(result.returncode)
