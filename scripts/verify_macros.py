"""Xác thực tên macro LaTeX sinh ra: chỉ được chứa chữ cái."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGETS = [
    Path(r"D:\IOT\report\generated\results_macros.tex"),
]

PATTERN = re.compile(r"\\newcommand\{\\([A-Za-z0-9_]+)\}")


def main() -> int:
    ok = True
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        names = PATTERN.findall(text)
        bad = sorted({n for n in names if re.search(r"[0-9_]", n)})
        dupes = sorted({n for n in names if names.count(n) > 1})
        print(f"{path.name}: {len(names)} macro, {len(set(names))} tên duy nhất")
        if bad:
            print("  LỖI tên chứa chữ số/gạch dưới:", bad)
            ok = False
        if dupes:
            print("  LỖI trùng tên:", dupes)
            ok = False
        invalid = sorted({n for n in names if not re.fullmatch(r"[A-Za-z]+", n)})
        if invalid:
            print("  tên không hợp lệ:", invalid)
            ok = False
        if len(names) != len(set(names)):
            ok = False

    # kiểm tra mọi macro được dùng trong báo cáo đều được định nghĩa
    report = Path(r"D:\IOT\report")
    used = set()
    files = [p for p in list(report.glob("*.tex")) + list((report / "sections").glob("*.tex"))
             if not p.name.startswith("_")]
    for tex in files:
        content = tex.read_text(encoding="utf-8")
        # chỉ khớp \R theo sau bởi chữ HOA — tránh bắt nhầm \RandomHorizontalFlip, \Rightarrow
        for m in re.findall(r"\\R[A-Z][A-Za-z]*", content):
            used.add(m[2:])
    defined = set()
    for path in TARGETS:
        # tên lệnh trong tệp là \R<Tên>, nên bỏ tiền tố \R khi so khớp với nơi dùng
        defined |= {n[1:] if n.startswith("R") else n
                    for n in PATTERN.findall(path.read_text(encoding="utf-8"))}
    missing = sorted(used - defined)
    if missing:
        print("CẢNH BÁO: macro dùng trong báo cáo nhưng chưa định nghĩa:", missing)
        ok = False
    unused = sorted(defined - used)
    if unused:
        print("(thông tin) macro định nghĩa nhưng chưa dùng:", unused)
    print("KẾT LUẬN:", "ĐẠT" if ok else "KHÔNG ĐẠT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
