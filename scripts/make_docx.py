"""Tạo bản Word (.docx) của báo cáo từ nguồn LaTeX.

Vì sao cần script riêng, không dùng `pandoc main-linux.tex` trực tiếp:

1. **Tham chiếu chéo.** Báo cáo dùng `\\ref{}` khắp nơi; pandoc đọc LaTeX không tự giải được số.
   Script đọc bảng `\\newlabel` trong `report-build/main-linux.aux` (do LaTeX sinh ra) để thay
   `\\ref{label}` bằng ĐÚNG số hiển thị trong PDF — nhờ vậy "Bảng 10", "Hình 9" trong Word khớp
   từng con số với bản PDF.
2. **`\\input` của bảng sinh tự động.** pandoc không theo `\\input`, nên phải nội suy 20 tệp
   `report/generated/*.tex` vào trước.
3. **TikZ.** pandoc không hiểu TikZ nên hình luồng huấn luyện sẽ biến mất; script thay bằng PNG
   do `scripts/make_tikz_figure.py` render (biên dịch bằng tectonic rồi đổ ra ảnh 300 DPI).
4. **Trang bìa.** Bìa có khung TikZ và logo; pandoc không dựng được. Script dựng bìa riêng bằng
   `python-docx` rồi ghép lên trước phần nội dung.

Quy trình:  LaTeX -> (tiền xử lý) -> merged.tex -> pandoc -> body.docx
            -> python-docx dựng bìa + ghép -> M8_<ten>.docx

Cách dùng:  python scripts/make_docx.py
Kết quả:    report-build/M8_DinhXuanHuy.docx
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
GEN = REPORT / "generated"
BUILD = ROOT / "report-build"
AUX = BUILD / "main-linux.aux"
PANDOC = ROOT / "tools" / "pandoc-3.5" / "bin" / "pandoc"
PIPELINE_PNG = ROOT / "results" / "figures" / "fig_pipeline_tikz.png"

SECTIONS = [
    "00_abstract", "01_introduction", "02_data", "03_method", "04_experiments",
    "05_results", "05b_results_extension", "06_discussion", "07_iot",
    "08_conclusion", "09_appendix",
]

REPO_URL = "https://github.com/dinhxuanhuy/iot-m8-multitask"


# --------------------------------------------------------------------------
# 1. Tham chiếu chéo từ .aux
# --------------------------------------------------------------------------
def load_refs() -> dict[str, str]:
    if not AUX.exists():
        print(f"  !! thiếu {AUX} — chạy tectonic với --keep-intermediates trước")
        return {}
    txt = AUX.read_text(encoding="utf-8", errors="ignore")
    refs: dict[str, str] = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", txt):
        refs[m.group(1)] = m.group(2)
    print(f"  đọc {len(refs)} nhãn từ .aux")
    return refs


def resolve_refs(tex: str, refs: dict[str, str]) -> str:
    """Thay \\ref{X} bằng số đã giải. Nhãn lạ thì để lại dạng đọc được."""
    def rep(m: re.Match) -> str:
        lab = m.group(1)
        num = refs.get(lab)
        if num is None:
            return "?"
        # số trang trong .aux có thể kèm \\relax hoặc lệnh khác
        num = re.sub(r"\\[a-zA-Z]+", "", num).strip()
        return num
    return re.sub(r"\\ref\{([^}]+)\}", rep, tex)


# --------------------------------------------------------------------------
# 2. Tiền xử lý LaTeX
# --------------------------------------------------------------------------
def _read_group(text: str, i: int) -> int:
    """``i`` trỏ vào ``{``. Trả về chỉ số NGAY SAU ``}`` khớp (đếm ngoặc lồng nhau)."""
    depth = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def collect_macros() -> str:
    """Gom mọi \\newcommand trong preamble của main-linux.tex và results_macros.tex.

    Cấu trúc một định nghĩa là ``\\newcommand{\\ten}[so-tham-so][mac-dinh]{than}``. Phải đọc
    LẦN LƯỢT từng nhóm ngoặc: nhóm tên, các nhóm ``[...]`` tuỳ chọn, rồi nhóm thân.

    Không dùng regex ``\\{.*?\\}`` vì thân macro có ngoặc lồng nhau
    (``\\textsf{\\textbf{#1}}``) nên khớp không tham lam sẽ dừng ở ``}`` đầu tiên, cắt cụt
    định nghĩa. Hệ quả: pandoc nhận được ``\\newcommand{\\cfg}`` không có thân, rồi báo
    ``unexpected E1`` ngay tại ``\\cfg{E1}`` đầu tiên trong bài.
    """
    def extract_defs(text: str) -> list[str]:
        out, i = [], 0
        key = "\\newcommand"
        while True:
            i = text.find(key, i)
            if i < 0:
                break
            j = i + len(key)
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if j >= len(text) or text[j] != "{":
                i = j
                continue
            j = _read_group(text, j)                       # nhóm tên
            while j < len(text) and text[j] == "[":        # [nargs] [default]
                k = text.find("]", j)
                j = k + 1 if k > 0 else j + 1
            if j < len(text) and text[j] == "{":
                j = _read_group(text, j)                   # nhóm thân
            out.append(text[i:j])
            i = j
        return out

    main = (REPORT / "main-linux.tex").read_text(encoding="utf-8")
    pre = main.split(r"\begin{document}")[0]
    defs = extract_defs(pre)
    defs.append((GEN / "results_macros.tex").read_text(encoding="utf-8"))
    print(f"  gom {len(defs)} khối định nghĩa macro")
    return "\n".join(defs)


def inline_inputs(tex: str) -> str:
    """Nội suy \\input{generated/X} bằng nội dung tệp tương ứng."""
    n = 0
    for _ in range(4):  # lặp để xử lý input lồng nhau
        def rep(m: re.Match) -> str:
            nonlocal n
            name = m.group(1).strip()
            p = REPORT / (name if name.endswith(".tex") else name + ".tex")
            if not p.exists():
                return ""
            n += 1
            return p.read_text(encoding="utf-8")
        tex, k = re.subn(r"\\input\{([^}]+)\}", rep, tex)
        if k == 0:
            break
    print(f"  nội suy {n} \\input")
    return tex


def transform(tex: str) -> str:
    # TikZ: thay bằng ảnh PNG đã render
    # (dùng lambda vì chuỗi thay thế chứa backslash, re.sub sẽ hiểu nhầm là escape)
    tikz = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.S)
    if PIPELINE_PNG.exists():
        img = f"\\includegraphics[width=0.85\\textwidth]{{{PIPELINE_PNG}}}"
        tex = tikz.sub(lambda m: img, tex, count=1)
    tex = tikz.sub("", tex)

    # hộp tự định nghĩa -> quote (pandoc hiểu)
    for env in ("keybox", "warnbox"):
        tex = tex.replace(f"\\begin{{{env}}}", "\\begin{quote}")
        tex = tex.replace(f"\\end{{{env}}}", "\\end{quote}")

    # bỏ các lệnh chỉ dành cho LaTeX
    for pat in (r"\\label\{[^}]*\}", r"\\justifying\b", r"\\vfill\b", r"\\clearpage\b",
                r"\\begin\{spacing\}\{[^}]*\}", r"\\end\{spacing\}", r"\\noindent\b",
                r"\\medskip\b", r"\\smallskip\b", r"\\bigskip\b", r"\\par\b"):
        tex = re.sub(pat, "", tex)

    # môi trường center quanh url -> đoạn thường
    tex = tex.replace("\\begin{center}", "").replace("\\end{center}", "")

    # \enquote{X} -> “X”
    tex = re.sub(r"\\enquote\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"“\1”", tex)

    # \code{X} -> \texttt{X} (pandoc hiểu)
    tex = re.sub(r"\\code\{", r"\\texttt{", tex)
    return tex


# --------------------------------------------------------------------------
# 3. pandoc -> docx
# --------------------------------------------------------------------------
def run_pandoc(merged: Path, out: Path) -> bool:
    if not PANDOC.exists():
        print(f"  !! không có pandoc tại {PANDOC}")
        return False
    r = subprocess.run(
        [str(PANDOC), str(merged), "-o", str(out), "--from", "latex", "--to", "docx",
         "--resource-path", f"{REPORT}:{ROOT}"],
        capture_output=True, text=True,
    )
    if not out.exists():
        print("  !! pandoc thất bại")
        print(r.stdout[-1200:])
        print(r.stderr[-2000:])
        return False
    print(f"  pandoc -> {out.name} ({out.stat().st_size/1024:.0f} KB)")
    return True


# --------------------------------------------------------------------------
# 4. Bìa + ghép
# --------------------------------------------------------------------------
def build_cover_and_merge(body: Path, out: Path) -> bool:
    """Dựng trang bìa NGAY TRONG tài liệu body rồi đẩy các phần tử bìa lên đầu.

    KHÔNG tạo Document mới rồi append các phần tử XML của body sang: ảnh trong .docx được
    tham chiếu qua relationship của TỪNG part, nên chuyển phần tử thô sang document khác sẽ
    làm ảnh mất quan hệ và biến mất (đã gặp: body có 11 ảnh, sau khi ghép còn 1 — chỉ còn
    logo của bìa). Cách đúng là thêm bìa vào chính tài liệu body, rồi hoán vị vị trí.
    """
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.shared import Pt, Cm, RGBColor

    doc = Document(str(body))

    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)

    n_before = len(doc.element.body)   # đánh dấu: mọi thứ thêm sau đây là nội dung bìa

    def para(text="", size=12, bold=False, italic=False, align="center",
             space_after=6, color=None):
        p = doc.add_paragraph()
        p.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER,
                       "left": WD_ALIGN_PARAGRAPH.LEFT,
                       "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
        p.paragraph_format.space_after = Pt(space_after)
        if text:
            r = p.add_run(text)
            r.font.size = Pt(size); r.bold = bold; r.italic = italic
            r.font.name = "Times New Roman"
            if color:
                r.font.color.rgb = RGBColor(*color)
        return p

    BLUE = (0x1F, 0x4E, 0x79)
    para("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT", 15, True, color=BLUE, space_after=0)
    para("THÀNH PHỐ HỒ CHÍ MINH", 15, True, color=BLUE, space_after=8)
    para("KHOA CÔNG NGHỆ THÔNG TIN", 12, True, space_after=14)

    logo = REPORT / "assets" / "hcmute-emblem.png"
    if logo.exists():
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(14)
        p.add_run().add_picture(str(logo), height=Cm(3.1))

    para("HỌC PHẦN: TRÍ TUỆ NHÂN TẠO CHO IoT", 13, True, space_after=14)
    para("BÁO CÁO TIỂU LUẬN CUỐI KHÓA", 14, True, space_after=14)
    for line in ("HỌC ĐA NHIỆM TRÊN CIFAR-10:",
                 "PHÂN LOẠI SIÊU LỚP KẾT HỢP",
                 "DỰ ĐOÁN GÓC XOAY ẢNH TỰ GIÁM SÁT"):
        para(line, 15, True, color=BLUE, space_after=2)
    para("Đánh giá trong điều kiện hạn chế nhãn huấn luyện", 12, italic=True, space_after=18)

    info = [("Mã đề tài:", "M8"), ("Sinh viên thực hiện:", "Đinh Xuân Huy"),
            ("Mã số sinh viên:", "23110102"), ("Phân nhóm:", "Đa nhiệm"),
            ("Dữ liệu:", "CIFAR-10"), ("Lớp học phần:", "Trí tuệ nhân tạo cho IoT"),
            ("Giảng viên hướng dẫn:", "Hồ Nhựt Minh"), ("Năm học:", "2026–2027")]
    t = doc.add_table(rows=0, cols=2)
    t.autofit = True
    for k, v in info:
        cells = t.add_row().cells
        r0 = cells[0].paragraphs[0].add_run(k); r0.bold = True; r0.font.size = Pt(12)
        r1 = cells[1].paragraphs[0].add_run(v); r1.font.size = Pt(12)

    para("", space_after=18)
    para("Mã nguồn:", 11, space_after=0)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(REPO_URL)
    r.font.size = Pt(11); r.font.color.rgb = RGBColor(0xC5, 0x5A, 0x11)
    r.font.name = "Times New Roman"

    para("", space_after=24)
    para("Thành phố Hồ Chí Minh, 2026", 12, True, space_after=0)

    sep = doc.add_paragraph(); sep.add_run().add_break(WD_BREAK.PAGE)

    # Hoán vị: đưa toàn bộ phần tử bìa (thêm sau n_before) lên đầu, giữ nguyên thứ tự.
    body_el = doc.element.body
    cover_els = list(body_el)[n_before:]
    for el in cover_els:
        body_el.remove(el)
    for i, el in enumerate(cover_els):
        body_el.insert(i, el)

    doc.save(str(out))
    import zipfile
    n_img = len([n for n in zipfile.ZipFile(out).namelist() if "media" in n])
    print(f"  -> {out.name} ({out.stat().st_size/1024:.0f} KB, {n_img} ảnh nhúng)")
    return True


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    refs = load_refs()

    body_tex = []
    for name in SECTIONS:
        p = REPORT / "sections" / f"{name}.tex"
        if p.exists():
            body_tex.append(p.read_text(encoding="utf-8"))
    tex = "\n".join(body_tex)
    tex = inline_inputs(tex)
    tex = transform(tex)
    tex = resolve_refs(tex, refs)

    merged = BUILD / "_merged_for_docx.tex"
    merged.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{amsmath,amssymb,graphicx,xcolor,textcomp}\n"
        "\\usepackage[utf8]{inputenc}\n"
        + collect_macros() + "\n\\begin{document}\n" + tex + "\n\\end{document}\n",
        encoding="utf-8")
    print(f"  merged: {merged.name} ({len(merged.read_text(encoding='utf-8'))/1024:.0f} KB)")

    body = BUILD / "_body.docx"
    if not run_pandoc(merged, body):
        return 1

    out = BUILD / "M8_DinhXuanHuy.docx"
    if not build_cover_and_merge(body, out):
        return 1
    print(f"\nHOÀN TẤT: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
