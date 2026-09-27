"""Đổi "## 8. Mở rộng contrastive" thành "## 10. Mở rộng contrastive" trong notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

NB = Path(r"D:\IOT\notebooks\M8_CIFAR10_MultiTask.ipynb")
OLD = "## 8. Mở rộng contrastive"
NEW = "## 10. Mở rộng contrastive"


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    n = 0
    for c in nb.cells:
        if c.cell_type == "markdown" and OLD in c.source:
            c.source = c.source.replace(OLD, NEW)
            n += 1
    nbformat.write(nb, str(NB))
    print("đã sửa", n, "ô")
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown":
            for line in c.source.splitlines():
                if line.startswith("##"):
                    print(f"  {i:3d} {line[:70]}")
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
