"""Render hình TikZ (luồng huấn luyện hai nhánh) thành PNG để nhúng vào file Word.

pandoc đọc LaTeX nhưng KHÔNG hiểu TikZ, nên hình vẽ bằng TikZ sẽ biến mất khi chuyển sang
.docx. Script này tách khối tikzpicture ra một tài liệu standalone, biên dịch bằng tectonic rồi
chuyển sang PNG, để bước tạo Word có ảnh thật thay cho hình vector.

Cách dùng:  python scripts/make_tikz_figure.py
Kết quả:    results/figures/fig_pipeline_tikz.png
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "sections" / "02_data.tex"
OUT_PNG = ROOT / "results" / "figures" / "fig_pipeline_tikz.png"
TMP = ROOT / "report-build" / "_tikz"

PREAMBLE = r"""\documentclass[border=8pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\usepackage{xcolor}
\definecolor{m8blue}{HTML}{1F4E79}
\definecolor{m8teal}{HTML}{0F7B6C}
\definecolor{m8orange}{HTML}{C55A11}
\definecolor{m8gray}{HTML}{595959}
\definecolor{m8light}{HTML}{F2F5F8}
\newcommand{\enc}{f_{\theta}}
\newcommand{\gsup}{g_{\mathrm{sup}}}
\newcommand{\grot}{g_{\mathrm{rot}}}
\newcommand{\DL}{D_{\mathrm{L}}}
\newcommand{\DU}{D_{\mathrm{U}}}
\newcommand{\Dtrain}{D_{\mathrm{train}}}
\newcommand{\Lsup}{L_{\mathrm{sup}}}
\newcommand{\Lrot}{L_{\mathrm{rot}}}
\newcommand{\code}[1]{\texttt{\small #1}}
\begin{document}
"""


def main() -> int:
    tex = SRC.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, re.S)
    if not m:
        print(f"  !! không tìm thấy tikzpicture trong {SRC}")
        return 1
    tikz = m.group(0)

    TMP.mkdir(parents=True, exist_ok=True)
    doc = TMP / "pipeline.tex"
    doc.write_text(PREAMBLE + tikz + "\n\\end{document}\n", encoding="utf-8")
    print(f"  tài liệu standalone: {doc}")

    tectonic = ROOT / "tools" / "tectonic"
    if not tectonic.exists():
        print(f"  !! không có {tectonic}")
        return 1

    r = subprocess.run(
        [str(tectonic), "-X", "compile", "pipeline.tex", "--outdir", "."],
        cwd=TMP, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "XDG_CACHE_HOME": str(ROOT / ".tectonic-cache"),
             "HOME": str(ROOT)},
    )
    pdf = TMP / "pipeline.pdf"
    if not pdf.exists():
        print("  !! biên dịch TikZ thất bại")
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        return 1
    print(f"  PDF: {pdf} ({pdf.stat().st_size} B)")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    r2 = subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(OUT_PNG.with_suffix(""))],
        capture_output=True, text=True,
    )
    if not OUT_PNG.exists():
        print("  !! chuyển PNG thất bại")
        print(r2.stderr[-800:])
        return 1
    print(f"  -> {OUT_PNG} ({OUT_PNG.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
