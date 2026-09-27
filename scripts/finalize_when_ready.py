"""Chờ khối thực nghiệm chính hoàn thành rồi tự động dựng lại báo cáo.

Chỉ cần cho tiện: kiểm tra số lượt đã xong, và khi đủ 27 lượt thì chạy
rotation diagnostic -> figures -> report data -> verify -> compile PDF.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CKPT = ROOT / "results" / "checkpoints"
EXPECTED = 27


def count_done() -> int:
    return len(list(CKPT.glob("*.json")))


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd or ROOT)).returncode


def main() -> int:
    print(f"Chờ tới khi đủ {EXPECTED} lượt kết quả...", flush=True)
    last = -1
    while True:
        n = count_done()
        if n != last:
            print(f"  đã xong {n}/{EXPECTED}", flush=True)
            last = n
        if n >= EXPECTED:
            break
        time.sleep(30)

    steps = [
        ([PY, "scripts/run_rotation_diagnostic.py"], None),
        ([PY, "scripts/make_figures.py"], None),
        ([PY, "scripts/make_report_data.py"], None),
        ([PY, "scripts/verify_macros.py"], None),
        ([PY, "scripts/verify_tables.py"], None),
        ([PY, "scripts/execute_notebook.py"], None),
        ([str(ROOT / "tools" / "tectonic.exe"), "-X", "compile", "main.tex"], ROOT / "report"),
    ]
    failed = []
    for cmd, cwd in steps:
        if run(cmd, cwd) != 0:
            failed.append(cmd[1] if len(cmd) > 1 else cmd[0])
    print("\n================ TỔNG KẾT ================")
    if failed:
        print("bước lỗi:", failed)
        return 1
    print("Hoàn tất. PDF:", ROOT / "report" / "main.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
