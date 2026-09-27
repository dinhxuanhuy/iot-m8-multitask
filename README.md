# M8 — Học đa nhiệm trên CIFAR-10 (Trí tuệ nhân tạo cho IoT)

**Sinh viên:** Đinh Xuân Huy — MSSV 23110102
**Đề tài:** Học đa nhiệm trên CIFAR-10: phân loại siêu lớp kết hợp dự đoán góc xoay ảnh tự giám sát.
**Câu hỏi nghiên cứu:** Nhiệm vụ phụ dự đoán góc xoay có cải thiện phân loại Animal/Vehicle khi
ngân sách nhãn bị giới hạn (10 %, 20 %, 50 %) hay không?

---

## 1. Kết quả bàn giao

| Sản phẩm | Đường dẫn |
|---|---|
| Báo cáo kỹ thuật (LaTeX) | [`report/main.tex`](report/main.tex) |
| Báo cáo kỹ thuật (PDF) | [`report/main.pdf`](report/main.pdf) |
| Notebook toàn quy trình | [`notebooks/M8_CIFAR10_MultiTask.ipynb`](notebooks/M8_CIFAR10_MultiTask.ipynb) |
| Notebook kèm output | `notebooks/M8_CIFAR10_MultiTask.executed.ipynb` |
| Mã nguồn thư viện | [`src/m8/`](src/m8) |
| Script chạy từng bước | [`scripts/`](scripts) |
| Kết quả thô + hình | [`results/`](results) |

---

## 2. Cấu trúc dự án

```
D:\IOT
├── .venv/                     # môi trường ảo Python 3.12 + PyTorch CUDA
├── requirements.txt
├── tools/tectonic.exe         # trình biên dịch LaTeX tự chứa (XeTeX)
├── data/
│   ├── cifar-10-batches-py/   # dữ liệu gốc (tự tải)
│   └── artifacts/splits.json  # chỉ mục split/subset đã đóng băng
├── src/m8/                    # thư viện dùng chung cho notebook và CLI
│   ├── config.py              # hằng số giao thức (seed, LR, epoch, lambda, ...)
│   ├── data.py                # ánh xạ siêu lớp, chia split, ngân sách nhãn, sinh rot90
│   ├── models.py              # ResNet-18 32x32 + MultiTaskNet hai đầu ra
│   ├── train.py               # vòng huấn luyện E1 / E2-L / E2
│   ├── evaluate.py            # Macro-F1, Accuracy, per-class, ma trận nhầm lẫn
│   ├── benchmark.py           # đếm tham số, dung lượng tệp, độ trễ mean/median/p95
│   └── experiments.py         # điều phối 27 lượt chạy
├── scripts/                   # chạy từng bước (xem mục 4)
├── notebooks/                 # notebook trình bày toàn quy trình
├── results/                   # kết quả thật: checkpoints, JSON, figures, logs
└── report/                    # báo cáo LaTeX -> PDF
```

---

## 3. Cài đặt môi trường

```powershell
cd D:\IOT
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**Bản PyTorch phải dùng là `2.11.0+cu128`.** Máy này bật *Smart App Control* của Windows, và
bản PyTorch CUDA mới hơn (2.12+) bị chặn nạp DLL với lỗi
`WinError 4551: An Application Control policy has blocked this file`. Bản `2.11.0+cu128` được
phép chạy và hỗ trợ đầy đủ kiến trúc `sm_120` của RTX 5070 Laptop:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 `
    torch==2.11.0 torchvision==0.26.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name m8-venv `
    --display-name "Python 3 (m8-venv)"
```

Kiểm tra nhanh:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

---

## 4. Chạy lại từ đầu

```powershell
# Cách 1: chạy tất cả trong một lệnh
.\.venv\Scripts\python.exe scripts\run_all.py

# Cách 2: chạy từng bước
.\.venv\Scripts\python.exe scripts\prepare_data.py            # tải CIFAR-10 + đóng băng split
.\.venv\Scripts\python.exe scripts\self_test.py               # kiểm tra kỹ thuật tự động
.\.venv\Scripts\python.exe scripts\run_experiments.py         # 27 lượt huấn luyện (100 epoch)
.\.venv\Scripts\python.exe scripts\check_contrastive.py       # kiểm tra nhánh contrastive
.\.venv\Scripts\python.exe scripts\sweep_lambda_c.py --epochs 40   # khóa λ_c (CHỈ validation)
.\.venv\Scripts\python.exe scripts\run_extensions.py          # khối mở rộng E3–E6
.\.venv\Scripts\python.exe scripts\run_inference_benchmark.py  # tham số, dung lượng, độ trễ
.\.venv\Scripts\python.exe scripts\run_rotation_diagnostic.py  # Rotation Accuracy chẩn đoán
.\.venv\Scripts\python.exe scripts\run_rotation_data_cost.py   # chi phí dựng view nhánh xoay
.\.venv\Scripts\python.exe scripts\make_dataset_figures.py     # hình minh họa dữ liệu
.\.venv\Scripts\python.exe scripts\make_figures.py             # hình cho báo cáo
.\.venv\Scripts\python.exe scripts\make_extension_figures.py   # hình khối mở rộng
.\.venv\Scripts\python.exe scripts\make_report_data.py         # sinh bảng + macro LaTeX
.\.venv\Scripts\python.exe scripts\verify_macros.py            # kiểm tra macro
.\.venv\Scripts\python.exe scripts\verify_tables.py            # kiểm tra bảng
cd report; ..\tools\tectonic.exe -X compile main.tex           # biên dịch PDF
.\.venv\Scripts\python.exe scripts\verify_pdf.py               # kiểm tra PDF
```

Chạy nhanh để kiểm tra đường ống (không dùng cho kết quả báo cáo):

```powershell
.\.venv\Scripts\python.exe scripts\run_all.py --epochs 12 --seeds 42
```

Kết quả mỗi lượt được ghi ngay vào `results/checkpoints/<tag>.json`, nên có thể dừng giữa
chừng và chạy lại — những lượt đã xong sẽ được bỏ qua.

---

## 5. Mở notebook

```powershell
.\.venv\Scripts\jupyter-lab.exe notebooks\M8_CIFAR10_MultiTask.ipynb
```

Hoặc thực thi không cần mở giao diện:

```powershell
.\.venv\Scripts\python.exe scripts\execute_notebook.py
```

---

## 6. Thiết kế thực nghiệm (bám đề cương)

| Thành phần | Thiết lập |
|---|---|
| Backbone | ResNet-18 khởi tạo ngẫu nhiên, không tiền huấn luyện, điều chỉnh cho ảnh 32×32 |
| Đầu ra | Superclass Head 512→2, Rotation Head 512→4 |
| Cấu hình | E1 (baseline), E2-L (chỉ D_L cho rotation), E2 (D_train cho rotation) |
| **Mở rộng** | **E3** (NT-Xent), **E4** (SupCon), **E5** (rotation + NT-Xent), **E6** (rotation + SupCon) |
| Mức nhãn | 10 % (4 500), 20 % (9 000), 50 % (22 500) — trên 45 000 ảnh D_train |
| Lần lặp | 42, 43, 44 (thay đổi subset nhãn **và** ngẫu nhiên huấn luyện) |
| Ngân sách | 100 epoch, 1 epoch = 1 lượt duyệt D_L, không Early Stopping |
| Optimizer | AdamW, lr 3×10⁻⁴, weight decay 10⁻⁴, betas (0.9, 0.999), eps 10⁻⁸ |
| Scheduler | CosineAnnealingLR, T_max = 100, eta_min = 0 |
| λ_rot | 0,5 cho thí nghiệm chính |
| Chỉ số chính | Macro-F1 trên 10 000 ảnh test |
| Chọn checkpoint | Macro-F1 validation cao nhất, hòa thì loss thấp hơn rồi epoch sớm hơn |

Tổng khối chính: 3 × 3 × 3 = **27 lượt huấn luyện**.

---

## 7. Cam kết về tính trung thực của số liệu

* Mọi con số trong báo cáo (Macro-F1, Accuracy, precision/recall/F1, ma trận nhầm lẫn, thời
  gian huấn luyện, số tham số, độ trễ) đều được sinh từ lần chạy thật, đọc từ `results/`.
* Bảng và macro LaTeX được **sinh tự động** bằng `scripts/make_report_data.py`; không có giá
  trị nào được nhập tay. Sửa tay `report/generated/*` sẽ bị ghi đè.
* Không có số liệu minh họa được đặt vào vị trí kết quả thực nghiệm.
* Kết quả âm hoặc không cải thiện được giữ nguyên và phân tích, không bị loại bỏ.
* Ba lần lặp chỉ cung cấp bằng chứng ban đầu về độ ổn định; báo cáo không đưa ra tuyên bố về
  ý nghĩa thống kê.

---

## 8. Ghi chú kỹ thuật đáng lưu ý

* **Tên lệnh LaTeX không được chứa chữ số.** `\newcommand{\RLatP95Cpu}` gây lỗi
  `Missing \begin{document}` rất khó đoán nguyên nhân; `scripts/verify_macros.py` kiểm tra
  điều này trước khi biên dịch.
* **Font thân bài phải có glyph tiếng Việt dựng sẵn.** Cambria không có, khiến TeX ghép dấu tổ hợp
  và *trích xuất văn bản từ PDF bị sai ký tự* (ví dụ `vấn` → `vaቷn`). Báo cáo dùng
  **Times New Roman** cho thân bài và Cambria Math cho công thức.
* **`\enquote` cần `style=` hợp lệ.** `csquotes` không có style `vietnamese`; nếu không chỉ định
  style, `\enquote` in ra dấu `?`. Báo cáo dùng `style=english`.
* **Khoảng trắng khi trích xuất PDF.** Một số trình đọc (ví dụ pymupdf) không chèn khoảng trắng sau
  glyph tiếng Việt; khoảng trắng **có tồn tại** ở tầng content stream. `scripts/verify_pdf.py`
  chuẩn hóa lại theo khoảng cách glyph để kiểm tra đúng.
* **Hai nhánh dùng các view khác nhau.** Nhánh phân loại dùng `RandomCrop` + `RandomHorizontalFlip`;
  nhánh xoay dùng ảnh gốc 32×32 rồi `torch.rot90`. Không thêm phép xoay/lật nào sau bước gán k.
* **Bộ sinh số ngẫu nhiên độc lập cho nhánh phụ.** Nếu dùng chung luồng RNG, việc thêm nhánh phụ
  sẽ dịch chuyển chuỗi augmentation của nhánh chính và làm chênh lệch ghép cặp Δ_s trở thành nhiễu.
* **E1 không chạy nhánh xoay, kể cả forward.** Đặt λ_rot = 0 mà vẫn đưa ảnh xoay qua BatchNorm
  vẫn làm thay đổi running statistics, nên không được coi là baseline. Kiểm thử
  `test_lambda_zero_still_changes_batchnorm_statistics` chứng minh điều này.
* **Dữ liệu CIFAR-10** được tải qua `torchvision.datasets.CIFAR10(download=True)`. Nếu mạng chậm,
  có thể tải thủ công `cifar-10-python.tar.gz` (md5 `c58f30108f718f92721af3b95e74349a`) vào
  thư mục `data/` rồi giải nén; torchvision sẽ tự bỏ qua bước tải.

---

## 9. Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q          # 33 bài kiểm thử
.\.venv\Scripts\python.exe scripts\verify_macros.py    # tên macro LaTeX hợp lệ
.\.venv\Scripts\python.exe scripts\verify_tables.py    # macro trong bảng đều được định nghĩa
.\.venv\Scripts\python.exe scripts\verify_pdf.py       # nội dung + glyph + Overfull trong PDF
```
