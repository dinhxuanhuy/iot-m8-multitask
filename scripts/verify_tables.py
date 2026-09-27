"""Kiểm tra mọi macro và nhãn được dùng trong các bảng sinh tự động đều tồn tại.

Chạy sau ``make_report_data.py`` để phát hiện sớm lỗi kiểu
"Undefined control sequence" trước khi biên dịch LaTeX.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPORT = Path(r"D:\IOT\report")
GEN = REPORT / "generated"

MACRO_DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}")


def main() -> int:
    macros_text = (GEN / "results_macros.tex").read_text(encoding="utf-8")
    defined = set(MACRO_DEF.findall(macros_text))

    # các lệnh do người soạn định nghĩa trong main.tex
    main_text = (REPORT / "main.tex").read_text(encoding="utf-8")
    defined |= set(MACRO_DEF.findall(main_text))

    # lệnh của gói LaTeX được phép xuất hiện trong bảng sinh tự động
    allowed = {
        "toprule", "midrule", "bottomrule", "cmidrule", "multicolumn", "multirow",
        "makecell", "textbf", "textit", "emph", "small", "footnotesize", "centering",
        "caption", "label", "ref", "cfg", "code", "textcolor", "enquote", "num",
        "si", "SI", "qty", "ang", "tabref", "figref", "secref", "eqrefv", "hline",
        "begin", "end", "rule", "par", "vspace", "hspace", "quad", "qquad", "text",
        "mathrm", "mathbb", "argmax", "times", "approx", "pm", "cdot", "to", "in",
        "varnothing", "setminus", "cup", "cap", "subset", "DL", "DU", "Dtrain",
        "Dval", "Dtest", "Lsup", "Lrot", "lamrot", "lam", "enc", "gsup", "grot",
        "norm", "dotfill", "smallskip", "medskip", "sffamily", "bfseries", "itshape",
        "RIncomplete", "addlinespace", "endfirsthead", "endhead", "endfoot",
        "caption", "color", "textwidth", "displaystyle", "mathcal", "left", "right",
        "texttt", "eta", "max", "min", "Delta", "setlength", "tabcolsep", "star",
        "exp", "log", "frac", "sum", "prod", "int", "operatorname", "underset",
        "overset", "boldsymbol", "ell", "mathbb", "mathbf", "mathrm", "in", "notin",
        "leq", "geq", "neq", "ll", "gg", "infty", "partial", "nabla", "tag",
        "label", "nonumber", "textwidth", "linewidth", "arraystretch", "renewcommand",
        "scriptsize", "footnotesize", "tiny", "small", "normalsize", "large", "Large",
        "begingroup", "endgroup", "c", "t", "enspace", "quad", "qquad", "hspace", "vspace",
        "lambda", "bullet", "tau", "alpha", "beta", "gamma", "sigma", "mu", "theta",
        "mathcal", "widehat", "lVert", "rVert", "top", "sim", "propto", "forall", "exists",
        "mathbb", "mathbf", "mathit", "mathsf", "binom", "sqrt", "text", "nm",
    }

    problems = []
    for path in sorted(GEN.glob("tab_*.tex")):
        text = path.read_text(encoding="utf-8")
        for macro in set(re.findall(r"\\([A-Za-z]+)", text)):
            if macro in allowed or macro in defined:
                continue
            problems.append(f"{path.name}: \\{macro}")

    if problems:
        print("THIẾU ĐỊNH NGHĨA:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 1
    print(f"OK: mọi macro trong {len(list(GEN.glob('tab_*.tex')))} bảng đều được định nghĩa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
