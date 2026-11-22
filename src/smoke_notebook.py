"""Chạy toàn bộ code cell của notebook như một smoke test (dùng file CSV chỉ định)."""
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("VIETNAM_RAW", str(Path(sys.argv[1]).resolve()))
os.environ.setdefault("IPYTHONDIR", "/tmp/ipython-smoke")

nb = json.loads(Path(sys.argv[2]).read_text())
ns = {"__name__": "__main__"}
failed = 0

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    print(f"\n===== cell {i} =====")
    try:
        exec(compile(src, f"<cell{i}>", "exec"), ns)
    except Exception:
        failed += 1
        traceback.print_exc()
        break

print("\nFAILED" if failed else "\nSMOKE TEST PASSED: mọi code cell chạy hết.")
sys.exit(1 if failed else 0)
