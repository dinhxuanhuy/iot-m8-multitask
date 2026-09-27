"""Cập nhật phần giới thiệu của notebook M8 (thêm mục contrastive)."""

from __future__ import annotations

from pathlib import Path

import nbformat

NB = Path(r"D:\IOT\notebooks\M8_CIFAR10_MultiTask.ipynb")

NEW_BLOCK = ("8. Hành vi hội tụ và chi phí suy luận\n"
             "9. Kết luận tự động từ kết quả\n"
             "10. Mở rộng contrastive: NT-Xent và SupCon (E3–E6)\n"
             "11. Ghi chú về tái lập và giới hạn\n")


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    n = 0
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("# M8"):
            src = c.source
            if "8. Chi phí suy luận" in src and "10. Mở rộng contrastive" not in src:
                src = src.replace("8. Chi phí suy luận và độ trễ\n", "")
                src = src.replace(OLD_BLOCK, NEW_BLOCK)
                c.source = src
                n += 1
    nbformat.write(nb, str(NB))
    print("đã cập nhật", n, "ô giới thiệu")
    print(nb.cells[0].source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
