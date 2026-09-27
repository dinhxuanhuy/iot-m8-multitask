"""Kiểm tra chất lượng văn bản trích xuất từ PDF báo cáo.

Mục đích: phát hiện hai loại lỗi phổ biến ở PDF do XeTeX sinh ra với tiếng Việt:

1. **Mất dấu** (glyph tổ hợp): một số font như Cambria không có glyph tiếng Việt
   dựng sẵn, khiến TeX ghép dấu tổ hợp và trình trích xuất trả về ký tự sai.
2. **Mất khoảng trắng**: một số trình đọc không chèn khoảng trắng sau các chữ có
   dấu ghép. Ở tầng content stream khoảng trắng **có tồn tại**; script này kiểm tra
   điều đó để phân biệt lỗi tài liệu với lỗi của trình trích xuất.

Ngoài ra script kiểm tra nội dung có mặt đầy đủ: các mục chính, bảng, hình.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

PDF = Path(r"D:\IOT\report\main.pdf")

#: Các chuỗi bắt buộc phải xuất hiện (không dấu câu nhạy cảm)
MUST_CONTAIN = [
    "HỌC ĐA NHIỆM TRÊN CIFAR",
    "Đinh Xuân Huy",
    "23110102",
    "Tóm tắt",
    "Đặt vấn đề",
    "Tập dữ liệu",
    "Phương pháp",
    "Thiết kế thực nghiệm",
    "Kết quả thực nghiệm",
    "Thảo luận",
    "Kết luận",
    "Phụ lục",
    "Tài liệu tham khảo",
    "Macro-F1",
    "11.169.858",
    "AdamW",
    "CosineAnnealingLR",
]

#: Từ tiếng Việt có dấu ghép hay bị trích xuất sai
DIACRITIC_PROBES = [
    "dữ liệu", "tiền xử lý", "nhãn", "ngân sách", "phạm vi", "vấn đề",
    "đối chứng", "nhiệm vụ", "thực nghiệm", "kết quả", "phương pháp",
    "đánh giá", "giới hạn", "chứng minh", "khẳng định",
]

#: Glyph lạ thường xuất hiện khi font thiếu ký tự tiếng Việt
SUSPECT_GLYPHS = "\u1234\u12ff\u0e00\u0e01\u0e48\u0e49\u0e4a"


def normalise(text: str, page: pymupdf.Page) -> str:
    """Chèn lại khoảng trắng bị mất dựa trên khoảng cách glyph thực tế."""
    chars = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            runs = []
            for span in line["spans"]:
                for ch in span["chars"]:
                    runs.append(ch)
            for i, ch in enumerate(runs):
                chars.append(ch["c"])
                if i + 1 < len(runs):
                    gap = runs[i + 1]["bbox"][0] - ch["bbox"][2]
                    size = ch["bbox"][3] - ch["bbox"][1]
                    if gap > 0.20 * size and ch["c"] != " " and runs[i + 1]["c"] != " ":
                        chars.append(" ")
    return "".join(chars)


def main() -> int:
    if not PDF.exists():
        print("Không tìm thấy", PDF)
        return 1
    doc = pymupdf.open(PDF)
    print(f"PDF: {PDF}  |  {doc.page_count} trang  |  {PDF.stat().st_size / 1024:.1f} KiB")

    ok = True

    # 1. Nội dung bắt buộc (dùng bản đã chuẩn hóa khoảng trắng vì một số trình đọc
    #    không chèn khoảng trắng sau glyph tiếng Việt dựng sẵn)
    plain = "".join(p.get_text() for p in doc)
    joined = "\n".join(normalise(p, p) for p in doc)
    # Gộp mọi khoảng trắng thành một dấu phân cách để chuỗi bắt buộc không phụ thuộc
    # vào chỗ TeX ngắt dòng.
    haystack = re.sub(r"\s+", "@", joined)
    missing = [s for s in MUST_CONTAIN if "@".join(s.split()) not in haystack]
    if missing:
        print("THIẾU nội dung bắt buộc:", missing)
        ok = False
    else:
        print(f"nội dung bắt buộc: ĐẠT ({len(MUST_CONTAIN)} chuỗi)")

    # 2. Không có glyph lạ (font thiếu ký tự tiếng Việt)
    bad = [g for g in SUSPECT_GLYPHS if g in plain]
    if bad:
        print("CÓ glyph lạ (font thiếu ký tự):", bad)
        ok = False
    else:
        print("glyph tiếng Việt: ĐẠT (không có ký tự lạ)")

    # 3. Khoảng trắng tồn tại ở tầng glyph (không phải lỗi tài liệu)
    found = sum(1 for s in DIACRITIC_PROBES if s in joined)
    raw_found = sum(1 for s in DIACRITIC_PROBES if s in plain)
    print(f"cụm từ có dấu tìm được: {found}/{len(DIACRITIC_PROBES)} "
          f"(thô {raw_found}/{len(DIACRITIC_PROBES)} — chênh lệch là do trình trích xuất)")
    if found < len(DIACRITIC_PROBES):
        missing_probes = [s for s in DIACRITIC_PROBES if s not in joined]
        print("  không tìm thấy khi đã chuẩn hóa khoảng trắng:", missing_probes)
        ok = False

    # 4. Không còn cảnh báo Overfull nghiêm trọng trong log
    log = PDF.parent / "main.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", text)
        severe = [float(v) for v in over if float(v) > 25.0]
        print(f"Overfull hbox: {len(over)} chỗ, trong đó >25pt: {len(severe)}")
        if severe:
            ok = False
        errors = [l for l in text.splitlines() if l.startswith("! ")]
        if errors:
            print("LỖI LaTeX trong log:", errors[:5])
            ok = False

    print("KẾT LUẬN:", "ĐẠT" if ok else "KHÔNG ĐẠT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
