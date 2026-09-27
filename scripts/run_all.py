"""Chạy toàn bộ quy trình M8 từ dữ liệu gốc tới báo cáo PDF.

Thứ tự thực hiện:
  1. prepare_data.py            — tải CIFAR-10, đóng băng split, kiểm tra chống rò rỉ nhãn
  2. self_test.py               — kiểm tra kỹ thuật tự động (shape, số tham số, rot90)
  3. run_experiments.py         — 27 lượt huấn luyện (3 cấu hình x 3 mức nhãn x 3 lần lặp)
  4. run_inference_benchmark.py — số tham số, dung lượng, độ trễ CPU/GPU
  5. run_rotation_diagnostic.py — Rotation Accuracy chẩn đoán trên validation
  6. make_figures.py            — sinh hình PNG cho báo cáo
  7. make_report_data.py        — sinh bảng và macro LaTeX
  8. verify_macros.py / verify_tables.py — kiểm tra macro trước khi biên dịch
  9. tectonic -X compile main.tex        — biên dịch báo cáo PDF

Ví dụ::

    .\\.venv\\Scripts\\python.exe scripts\\run_all.py
    .\\.venv\\Scripts\\python.exe scripts\\run_all.py --epochs 20 --seeds 42   # chạy nhanh
    .\\.venv\\Scripts\\python.exe scripts\\run_all.py --skip-train             # chỉ dựng báo cáo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
REPORT = ROOT / "report"


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT))
    if proc.returncode != 0:
        print(f"!!! lệnh trả mã thoát {proc.returncode}", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Quy trình đầy đủ của đề tài M8")
    ap.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch (chạy nhanh)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true", help="bỏ qua các bước huấn luyện")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="bỏ qua khảo sát λ_c (dùng giá trị đã khóa trước đó)")
    ap.add_argument("--sweep-epochs", type=int, default=40,
                    help="số epoch cho khảo sát λ_c (mặc định 40)")
    ap.add_argument("--skip-notebook", action="store_true", help="không thực thi notebook")
    ap.add_argument("--no-compile", action="store_true", help="không biên dịch LaTeX")
    args = ap.parse_args()

    failed: list[str] = []

    def step(name: str, cmd: list[str], cwd: Path | None = None) -> None:
        if run(cmd, cwd) != 0:
            failed.append(name)

    if not args.skip_train:
        step("prepare_data", [PY, "scripts/prepare_data.py"])
        step("self_test", [PY, "scripts/self_test.py"])
        train_cmd = [PY, "scripts/run_experiments.py"]
        if args.epochs:
            train_cmd += ["--epochs", str(args.epochs)]
        if args.seeds:
            train_cmd += ["--seeds", *[str(s) for s in args.seeds]]
        step("run_experiments", train_cmd)
        # Phần mở rộng contrastive: kiểm tra -> khóa λ_c bằng validation -> chạy E3--E6
        step("check_contrastive", [PY, "scripts/check_contrastive.py"])
        if not args.skip_sweep:
            step("sweep_lambda_c", [PY, "scripts/sweep_lambda_c.py",
                                    "--epochs", str(args.sweep_epochs)])
            step("lock_lambda_c", [PY, "scripts/lock_lambda_c.py"])
        ext_cmd = [PY, "scripts/run_extensions.py"]
        if args.epochs:
            ext_cmd += ["--epochs", str(args.epochs)]
        if args.seeds:
            ext_cmd += ["--seeds", *[str(s) for s in args.seeds]]
        step("run_extensions", ext_cmd)

    step("inference_benchmark", [PY, "scripts/run_inference_benchmark.py"])
    step("rotation_diagnostic", [PY, "scripts/run_rotation_diagnostic.py"])
    step("rotation_data_cost", [PY, "scripts/run_rotation_data_cost.py"])
    step("make_dataset_figures", [PY, "scripts/make_dataset_figures.py"])
    step("make_figures", [PY, "scripts/make_figures.py"])
    step("make_extension_figures", [PY, "scripts/make_extension_figures.py"])
    step("make_report_data", [PY, "scripts/make_report_data.py"])
    step("verify_macros", [PY, "scripts/verify_macros.py"])
    step("verify_tables", [PY, "scripts/verify_tables.py"])

    if not args.skip_notebook:
        step(
            "notebook",
            [
                str(ROOT / ".venv" / "Scripts" / "jupyter-nbconvert.exe"),
                "--to", "notebook", "--execute",
                "--ExecutePreprocessor.kernel_name=m8-venv",
                "--ExecutePreprocessor.timeout=3600",
                "--output", "M8_CIFAR10_MultiTask.executed.ipynb",
                "notebooks/M8_CIFAR10_MultiTask.ipynb",
            ],
        )

    if not args.no_compile:
        tectonic = ROOT / "tools" / "tectonic.exe"
        if tectonic.exists():
            step("latex", [str(tectonic), "-X", "compile", "main.tex"], cwd=REPORT)
        else:
            step("latex", ["latexmk", "-pdf", "main.tex"], cwd=REPORT)
        step("verify_pdf", [PY, "scripts/verify_pdf.py"])

    print("\n================ TỔNG KẾT ================")
    if failed:
        print("Bước lỗi:", ", ".join(failed))
        return 1
    print("Tất cả các bước hoàn thành.")
    print("Báo cáo:", REPORT / "main.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
