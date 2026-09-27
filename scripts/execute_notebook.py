"""Thực thi notebook M8 và lưu bản có output.

Dùng ``allow_errors=False`` để lỗi nào cũng làm script thất bại (đúng tinh thần
"kiểm tra trước khi tuyên bố hoàn thành"), nhưng in rõ ô nào lỗi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(r"D:\IOT\notebooks\M8_CIFAR10_MultiTask.ipynb")
OUT = Path(r"D:\IOT\notebooks\M8_CIFAR10_MultiTask.executed.ipynb")


def main() -> int:
    nb = nbformat.read(NB, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="m8-venv", allow_errors=True)
    client.execute()
    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename"), (out.get("evalue") or "")[:200]))
    nbformat.write(nb, OUT)
    print(f"Đã ghi {OUT}")
    if errors:
        print(f"CÓ {len(errors)} ô lỗi:")
        for i, name, val in errors:
            print(f"  ô {i}: {name}: {val}")
        return 1
    print("Không có ô nào lỗi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
