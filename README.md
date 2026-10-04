# M8 — Học đa nhiệm trên CIFAR-10 (Trí tuệ nhân tạo cho IoT)

**Sinh viên:** Đinh Xuân Huy — MSSV 23110102
**Đề tài:** Học đa nhiệm trên CIFAR-10: phân loại siêu lớp kết hợp dự đoán góc xoay ảnh tự giám sát.
**Câu hỏi nghiên cứu:** Nhiệm vụ phụ dự đoán góc xoay có cải thiện phân loại Animal/Vehicle khi
ngân sách nhãn bị giới hạn (10 %, 20 %, 50 %) hay không?

---

## 📄 Báo cáo

> ### 👉 [**Đọc báo cáo đầy đủ — `report/main.pdf` (47 trang)**](report/main.pdf)
>
> Báo cáo kỹ thuật hoàn chỉnh: đặt vấn đề, dữ liệu, phương pháp, giao thức, kết quả khối chính
> (E1/E2-L/E2) và khối mở rộng contrastive (E3–E6), chẩn đoán Rotation Accuracy, chi phí huấn
> luyện/suy luận, phân tích lỗi và giới hạn.
>
> **Mã nguồn LaTeX:** [`report/main.tex`](report/main.tex) (Windows · Times New Roman) ·
> [`report/main-linux.tex`](report/main-linux.tex) (Linux · Liberation + font đi kèm).
> `report/main.pdf` hiện là bản build từ `main-linux.tex`.

### Kết quả chính (Macro-F1 trên 10.000 ảnh test, mức 10 % nhãn, trung bình 3 lần lặp)

| Cấu hình | Macro-F1 | Δ ghép cặp so với E1 | Số lần lặp dương |
|---|---|---|---|
| **E6** xoay + SupCon | **95,79 ± 0,10** | **+2,60 ± 0,25 pp** | 3/3 |
| E4 chỉ SupCon | 95,27 ± 0,22 | +2,07 ± 0,36 pp | 3/3 |
| E5 xoay + NT-Xent | 94,96 ± 0,07 | +1,77 ± 0,34 pp | 3/3 |
| E2 xoay + D_train | 94,70 ± 0,34 | +1,51 ± 0,06 pp | 3/3 |
| E3 chỉ NT-Xent | 94,38 ± 0,22 | +1,19 ± 0,56 pp | 3/3 |
| E1 baseline | 93,19 ± 0,35 | — | — |
| E2-L xoay (chỉ D_L) | 93,13 ± 0,30 | −0,06 ± 0,38 pp | 1/3 |

Ba cấu hình *bắt buộc* của đề cương là **E1**, **E2-L**, **E2**; bốn cấu hình **E3–E6** thuộc
phần mở rộng tùy chọn (mục 3.3 đề cương) và được triển khai đầy đủ.

---

## 1. Kết quả bàn giao

| Sản phẩm | Đường dẫn |
|---|---|
| **Notebook cuối cùng (tổng hợp tất cả)** | [`notebooks/M8_Final_All.ipynb`](notebooks/M8_Final_All.ipynb) |
| Báo cáo kỹ thuật (LaTeX) | [`report/main.tex`](report/main.tex) (Windows) · [`report/main-linux.tex`](report/main-linux.tex) (Linux) |
| Báo cáo kỹ thuật (PDF) | [`report/main.pdf`](report/main.pdf) |

> **Về tệp `report/main.pdf`.** Tệp này là bản build **từ `main-linux.tex`** (XeLaTeX + font
> Liberation đi kèm trong `report/assets/fonts/`), vì môi trường soạn báo cáo là Linux.
> Nội dung giống hệt `main.tex`; chỉ khác bộ font. Trên Windows, chạy lại
> `cd report && tectonic -X compile main.tex` để có bản dùng Times New Roman như thiết kế gốc.
> Bản build đã kiểm tra đầy đủ nằm ở `report-build/BaoCao_M8_DinhXuanHuy_23110102.pdf`.
| Notebook toàn quy trình | [`notebooks/M8_CIFAR10_MultiTask.ipynb`](notebooks/M8_CIFAR10_MultiTask.ipynb) |
| Notebook kèm output | `notebooks/M8_CIFAR10_MultiTask.executed.ipynb` |
| Mã nguồn thư viện | [`src/m8/`](src/m8) |
| Script chạy từng bước | [`scripts/`](scripts) |
| Kết quả thô + hình | [`results/`](results) |
| **Chỉ mục split/subset đã đóng băng** | [`results/split_indices.json`](results/split_indices.json) (3 MB) |
| **Trọng số đã huấn luyện (2 checkpoint)** | [`E4_L10_s42.pt`](results/checkpoints_ext/E4_L10_s42.pt) 95,29 % · [`E3_L10_s42.pt`](results/checkpoints_ext/E3_L10_s42.pt) 94,13 % — xem [`README.md`](results/checkpoints_ext/README.md) |
| Kiểm tra che nhãn | [`results/benchmark_audit.json`](results/benchmark_audit.json) |
| Dự đoán test theo chỉ mục | `results/test_predictions/*.npz` |
| Tài liệu Colab CLI (tham chiếu) | [`vendor/colab-cli-docs/`](vendor/colab-cli-docs/INDEX.md) |

> **Notebook cuối cùng** `notebooks/M8_Final_All.ipynb` là bản tổng hợp duy nhất: môi trường, dữ
> liệu, kiến trúc, hàm mất mát, sáu cấu hình E1–E6, kết quả cả hai khối, chẩn đoán Rotation
> Accuracy, chi phí, hình, kết luận tự sinh và hướng dẫn tái lập trên Colab GPU.
> Sinh lại bằng `python scripts/make_final_notebook.py`; không sửa file `.ipynb` bằng tay vì
> sẽ bị ghi đè.

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

### 4.1. Biên dịch báo cáo trên Linux (`main-linux.tex`)

`report/main.tex` yêu cầu **Times New Roman / Segoe UI / Consolas / Cambria Math** — đều là
font Windows. Trên Linux, dùng `report/main-linux.tex`: cùng nội dung, chỉ đổi bộ font sang
**Liberation Serif + Liberation Sans + Liberation Mono + STIX Two Math**.

```bash
cd report
XDG_CACHE_HOME=../.tectonic-cache ../tools/tectonic -X compile main-linux.tex --outdir ../report-build
```

**Font được đóng gói kèm** trong `report/assets/fonts/` và nạp bằng `Path=`, nên biên dịch
được trên máy **không cài sẵn** các font đó. Đã kiểm chứng: tạm giấu một file font trong repo
thì build báo lỗi `The font "STIXTwoText-Regular" cannot be found`.

> ⚠️ **Phải `cd report` trước khi biên dịch.** `Path=` trong `main-linux.tex` là đường dẫn
> tương đối (`assets/fonts/...`), nên thư mục làm việc phải là `report/`.

Vì sao chọn Liberation Serif: đây là bản **tương thích metric của Times New Roman**, bám sát quy
định trình bày học thuật VÀ render dấu tiếng Việt tốt hơn STIX Two Text (xem mục 8)
Việt Nam (Times New Roman 13). Đã đối chiếu bảng `cmap` của từng file: đủ **91/91 ký tự có
dấu** dùng trong báo cáo ở cả 4 mặt chữ. Xem mục 8 để biết các font **không** dùng được.

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
Khối mở rộng: 4 × 1 × 3 = **12 lượt huấn luyện** (chỉ ở mức 10 % nhãn).

### 6.1 Kết quả cuối (Macro-F1 trên D_test, mức 10 % nhãn, trung bình 3 lần lặp)

| Cấu hình | Macro-F1 | Δ ghép cặp so với E1 | Số lần lặp dương |
|---|---|---|---|
| **E6** xoay + SupCon | **95,79 ± 0,10** | **+2,60 ± 0,25 pp** | 3/3 |
| E4 SupCon (chỉ D_L) | 95,27 ± 0,22 | +2,07 ± 0,36 pp | 3/3 |
| E5 xoay + NT-Xent | 94,96 ± 0,07 | +1,77 ± 0,34 pp | 3/3 |
| E2 xoay + D_train | 94,70 ± 0,34 | +1,51 ± 0,06 pp | 3/3 |
| E3 NT-Xent (D_train) | 94,38 ± 0,22 | +1,19 ± 0,56 pp | 3/3 |
| E1 baseline | 93,19 ± 0,35 | — | — |
| E2-L xoay (chỉ D_L) | 93,13 ± 0,30 | −0,06 ± 0,38 pp | 1/3 |

Hai so sánh có tính cấu trúc, đều 3/3 lần lặp dương:

* **SupCon > NT-Xent** ở cả hai điều kiện: E6 − E5 = +0,83 pp; E4 − E3 = +0,88 pp.
* **Nhánh xoay và contrastive bổ trợ nhau**: E6 − E4 = +0,52 pp; E5 − E3 = +0,58 pp —
  trong khi nhánh xoay chạy một mình (E2-L) không khác baseline.

Khối mở rộng chạy trên Google Colab L4 (máy local không có driver NVIDIA). Tổng thời gian
huấn luyện khối chính 8,26 giờ; mỗi lượt mở rộng ~25 phút trên L4.

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
  **Times New Roman** cho thân bài và Cambria Math cho công thức (bản Linux: STIX Two Text /
  STIX Two Math, xem mục 4.1).
* **`setspace` phải được dùng thật, không chỉ nạp.** Tiếng Việt có dấu chồng **hai tầng**
  (`ẫ ộ ự ỡ Ậ Ệ Ộ`). Để giãn dòng đơn, dấu của dòng dưới **đụng vào dòng trên và bị cắt cụt khi
  in** — trông như *mất dấu*, dù font có đủ glyph. Chuẩn học thuật Việt Nam quy định giãn dòng
  1,5; báo cáo dùng `\setstretch{1.25}`. Lưu ý `\setstretch` thuộc `setspace` nên **phải** đặt
  *sau* `\usepackage{setspace}`. Trang bìa được ép riêng về giãn dòng 1,0 (`spacing`
  environment) vì nếu để 1,25 thì bìa tràn sang trang 2.
* **Không dùng VARIABLE FONT với `xdvipdfmx`.** `Noto Serif`, `Noto Sans`, `Cantarell`,
  `Adwaita Sans` (bản cài trên Ubuntu) là variable font và làm build gãy với
  `Invalid font: -1`. Đây là lỗi rất khó đoán vì font vẫn hiện trong `fc-list`.
* **Một số font thiếu hẳn ký tự tiếng Việt.** `Caladea` thiếu 59, `Nimbus Roman` và `C059`
  thiếu 62 trong số 91 ký tự có dấu mà báo cáo dùng. Kiểm tra bằng cách đọc bảng `cmap` của
  file font, đối chiếu với danh sách ký tự trích từ `report/**/*.tex`.
* **Có đủ glyph CHƯA chắc đã render dấu đúng — phải kiểm bằng mắt.** Cả `Carlito` lẫn
  `STIX Two Text` đều **có đủ** 93/93 ký tự tiếng Việt trong `cmap`, nhưng vẫn render sai:
  dấu nặng (dấu chấm dưới) của `ộ ợ ự ậ ệ` được đặt **quá thấp và tách rời** khỏi con chữ, còn
  dấu chồng hai tầng ở `Ờ` thì chồng lên nhau. `Liberation Serif`/`Liberation Sans` đặt các dấu
  này bám sát chữ. Cách kiểm: `pdftoppm -r 600` rồi soi vùng chữ có dấu, **hoặc** render cùng
  một câu bằng `PIL.ImageFont` cho từng font rồi đặt cạnh nhau. Đừng tin `cmap` — hãy nhìn.
* **`\textsc` cần mặt chữ small-caps mà font thường KHÔNG có.** Cả `Liberation Serif` (bản Linux)
  lẫn `Times New Roman` (bản Windows) đều không có `smcp`, nên `\textsc{E1}` rơi vào shape
  `TU/LiberationSerif(0)/m/sc` không tồn tại: LaTeX thay bằng mặt chữ khác và cảnh báo
  `Font shape ... undefined`. Đã đổi `\cfg` sang `\textsf{\textbf{...}}` ở **cả hai** `main.tex`
  và `main-linux.tex`. (`STIX Two Text` thì *có* `smcp`, nên lỗi này chỉ lộ ra sau khi đổi font.)
* **`\input` một file chỉ có comment vẫn tính là `\input`.** `generated/tab_extension_results.tex`
  và `tab_extension_cost.tex` chỉ chứa comment, nên `\label` bên trong không tồn tại và
  `\ref` tới chúng in ra `??` trong PDF.
* **Ảnh CIFAR là 32×32 px — phải dùng `interpolation="nearest"`.** Nội suy làm mượt mặc định của
  `imshow` biến ảnh 32×32 thành nhoè khi phóng to. Ngoài ra hình lưới 10 hàng × 6 cột cho tỉ lệ
  0,53; nhúng ở `\textwidth` thành ~30 cm, VƯỢT chiều cao trang, khiến LaTeX **cắt mất hàng cuối,
  legend và cả caption**. `scripts/make_dataset_figures.py` dùng bố cục 5 hàng × 2 khối 6 cột
  (tỉ lệ ~2,2) và `savefig.dpi = 300`.
* **Kiểm tra DPI hiệu dụng của mọi hình trước khi nộp.** Ở bề rộng `\textwidth` (15,9 cm), hình cần
  ≥ ~1880 px để đạt 300 DPI. Script kiểm tra nhanh:
  `python -c "from PIL import Image; ..."` — xem `scripts/` hoặc mục 11 của notebook cuối.
  Với `savefig.dpi = 200`, nhiều hình chỉ đạt 155–289 DPI và bị mờ khi in.
* **`make_report_data.py` cần `results/checkpoints/`.** Thư mục này (~1,4 GB) không kèm trong kho,
  nên script báo "Chưa có kết quả" và KHÔNG sinh bảng. Trên máy có checkpoints thì chạy bình
  thường. `report-build/regen_extension_tables.py` là harness cục bộ tái tạo 27 lượt khối chính
  từ `results/summary.json` để sinh được bảng khi thiếu checkpoints; harness **bỏ qua**
  `tab_convergence` vì hàm đó tự gọi `load_all_runs(CHECKPOINT_DIR)` và sẽ trả về comment rỗng,
  phá bảng đúng đang có.
* **`\cfg{}` là `\textsc{}`, KHÔNG dùng được trong math mode.** Viết `L_{\cfg{E3}}` sẽ lỗi; trong
  công thức phải dùng `L^{(\mathrm{E3})}`.
* **Dấu âm trong bảng phải là `-` trần rồi bọc `$...$` một lần.** Bản cũ đặt
  `sign = "$-$"` rồi bọc thêm `$...$`, sinh ra `$$-$0,06$` — một cặp `$$` (display math rỗng)
  chèn vào giữa bảng.
* **Nhãn bảng phải phân biệt theo tham số.** `tab_per_class` và `tab_per_class_50` từng dùng
  chung `tab:per-class`, gây "multiply defined labels" và làm `\ref` trỏ không xác định.
* **Log huấn luyện trên Windows là UTF-16, không phải UTF-8.** Chuyển hướng output bằng
  PowerShell (`python ... > log.txt`) tạo tệp UTF-16LE, nên `grep`/`diff`/`less` trên Linux đọc
  ra ký tự rác và `grep` báo "binary file matches". Nội dung log vẫn đủ (loss từng nhánh, chỉ số
  validation, epoch được chọn, giây/epoch), chỉ sai encoding. Đã chuyển toàn bộ
  `results/logs/*.log` sang UTF-8; khi chạy lại trên Windows nên dùng
  `... | Out-File -Encoding utf8 log.txt` hoặc `Tee-Object`.
* **`data/` bị `.gitignore` nhưng đề cương vẫn yêu cầu bàn giao chỉ mục split.** Đề cương mục 5.2
  đòi *"danh sách chỉ mục split/subset"*, mà chỉ mục nằm ở `data/artifacts/splits.json` (đã bị
  bỏ qua cùng CIFAR-10 tải về). `scripts/prepare_data.py` nay sao chép thêm sang
  `results/split_indices.json` (3 MB) và `results/benchmark_audit.json`; bản trong `results/` là
  bản sao **chính xác** (kiểm bằng `md5sum`) của tệp mà mã nguồn dùng.
* **`build_macros` nhận `summary_all` nên `\RRuns`/`\RTotalHours` từng đếm cả hai khối.** Đây là
  lỗi **tiềm ẩn** chỉ lộ ra khi đã có kết quả mở rộng: văn xuôi mục 5.1 và 9.1 dùng hai macro này
  cho *khối chính*, nên báo cáo in ra "33 lượt" ngay cạnh câu "3 cấu hình × 3 mức nhãn × 3 lần
  lặp" (= 27). Đã lọc lại theo `MAIN_CONFIGS` bên trong `build_macros`. Các macro khác
  (`\RFindingC*`, `\REoneTenVal`…) không bị ảnh hưởng vì chúng tra cứu theo *tên cấu hình tường
  minh* thay vì theo toàn bộ danh sách lượt chạy.
* **Lệnh `colab exec` dài sẽ mất kết nối WebSocket** (`RuntimeError: Connection was lost`) sau
  khoảng một giờ, nhưng **tiến trình trên VM vẫn chạy tiếp và lưu kết quả đầy đủ**. Vì vậy nên
  (a) đặt `--timeout` lớn, và (b) đồng bộ kết quả về định kỳ bằng
  `.colab-bundle/sync_results.sh` thay vì tin vào việc job kết thúc sạch.
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
.\.venv\Scripts\python.exe -m pytest tests -q          # 49 bài kiểm thử
.\.venv\Scripts\python.exe scripts\verify_macros.py    # tên macro LaTeX hợp lệ
.\.venv\Scripts\python.exe scripts\verify_tables.py    # macro trong bảng đều được định nghĩa
.\.venv\Scripts\python.exe scripts\verify_pdf.py       # nội dung + glyph + Overfull trong PDF
```
