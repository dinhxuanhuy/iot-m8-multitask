"""Xuất **dự đoán test theo chỉ mục** cho một lượt chạy đã có checkpoint.

Yêu cầu của đề cương (mục 5.2): *"Đóng gói checkpoint được chọn bằng validation cho các lượt
chính; kèm logits hoặc dự đoán test theo chỉ mục để kiểm tra lại chỉ số."*

Script này đọc checkpoint ``results/checkpoints/<tag>.pt`` (hoặc ``checkpoints_ext/``), chạy suy
luận trên **toàn bộ 10.000 ảnh test** theo đúng thứ tự gốc của CIFAR-10, rồi ghi ra
``results/test_predictions/<tag>.npz`` gồm:

  * ``test_index`` — chỉ mục 0..9999 trong tập test chính thức (KHÔNG xáo trộn)
  * ``y_true``     — nhãn siêu lớp thật
  * ``y_pred``     — nhãn dự đoán
  * ``logits``     — logits hai chiều (N, 2)
  * ``prob``       — xác suất softmax (N, 2)

Đồng thời **kiểm tra tính nhất quán**: tính lại Macro-F1 và Accuracy từ chính các dự đoán vừa
xuất, so với chỉ số đã ghi trong ``<tag>.json``. Lệch quá 1e-9 là báo lỗi.

Cách dùng::

    python scripts/run_error_analysis.py                    # các lượt chính ở mức 10 %, seed 42
    python scripts/run_error_analysis.py --configs E1 E2-L E2 --fraction 0.10 --seed 42
    python scripts/run_error_analysis.py --tag E3_L10_s42 --ext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import (  # noqa: E402
    CHECKPOINT_DIR, MAIN_CONFIGS, RESULTS_DIR, RunSpec,
)
from m8.data import (  # noqa: E402
    SuperclassDataset, build_benchmark, eval_transform, get_cifar10,
)
from m8.evaluate import macro_f1  # noqa: E402
from m8.models import build_model  # noqa: E402

OUT_DIR = RESULTS_DIR / "test_predictions"


@torch.no_grad()
def predict_with_logits(model, loader, device) -> dict:
    """Suy luận, giữ nguyên thứ tự mẫu (loader phải ``shuffle=False``)."""
    model.eval()
    ys, ps, logits_all = [], [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        logits_all.append(logits.detach().cpu().numpy())
        ys.append(y.numpy())
        ps.append(logits.argmax(dim=1).detach().cpu().numpy())
    return {
        "y_true": np.concatenate(ys).astype(np.int64),
        "y_pred": np.concatenate(ps).astype(np.int64),
        "logits": np.concatenate(logits_all).astype(np.float32),
    }


def load_checkpoint(path: Path, device) -> tuple:
    blob = torch.load(path, map_location=device, weights_only=False)
    spec = blob["spec"]
    model = build_model(
        with_rotation_head=spec.get("rotation_source", "none") != "none",
        proj_dim=None,
    ).to(device)
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    if missing:
        # Projection Head chỉ có ở lượt mở rộng; thiếu nó là bình thường khi suy luận
        real = [k for k in missing if "projection_head" not in k]
        if real:
            raise RuntimeError(f"thiếu khoá trọng số: {real[:5]}")
    return model, spec


def main() -> int:
    ap = argparse.ArgumentParser(description="Xuất dự đoán test theo chỉ mục")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None, help="chỉ định thẳng một tag")
    ap.add_argument("--ext", action="store_true", help="đọc từ checkpoints_ext/")
    ap.add_argument("--max-flipped", type=int, default=2,
                    help="dung sai tính bằng SỐ ẢNH bị lật nhãn khi suy luận lại trên thiết bị "
                         "khác thiết bị huấn luyện (mặc định 2; CPU vs CUDA thường lật 0-2 ảnh "
                         "trên 10.000)")
    args = ap.parse_args()

    ckpt_dir = RESULTS_DIR / ("checkpoints_ext" if args.ext else "checkpoints")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.tag:
        tags = [args.tag]
    else:
        cfgs = args.configs or list(MAIN_CONFIGS)
        tags = [RunSpec(config=c, label_fraction=args.fraction, seed=args.seed).tag
                for c in cfgs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train, test = get_cifar10(download=False)
    bench = build_benchmark(train.targets)
    test_ds = SuperclassDataset(test, bench.test_idx, transform=eval_transform(),
                                return_superclass=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"D_test = {len(test_ds):,} ảnh, thứ tự gốc 0..{len(test_ds)-1} (không xáo trộn)")

    ok_all = True
    for tag in tags:
        ckpt = ckpt_dir / f"{tag}.pt"
        meta = ckpt_dir / f"{tag}.json"
        if not ckpt.exists():
            print(f"  !! không có {ckpt}")
            ok_all = False
            continue

        model, spec = load_checkpoint(ckpt, device)
        pred = predict_with_logits(model, test_loader, device)
        pred["test_index"] = np.arange(len(pred["y_true"]), dtype=np.int64)
        prob = torch.softmax(torch.from_numpy(pred["logits"]), dim=1).numpy().astype(np.float32)
        pred["prob"] = prob

        out = OUT_DIR / f"{tag}.npz"
        np.savez_compressed(out, **pred)

        # ---- kiểm tra nhất quán với chỉ số đã báo cáo ----
        # LƯU Ý: suy luận lại trên thiết bị KHÁC thiết bị đã huấn luyện (CPU vs CUDA) có thể
        # lật nhãn vài ảnh do làm tròn dấu phẩy động. Vì vậy dung sai được đo bằng SỐ ẢNH BỊ
        # LẬT, không bằng đơn vị Macro-F1: một ngưỡng F1 cố định là đại lượng phụ thuộc thiết
        # bị và không diễn giải được. Ở đây |ΔAcc| * N chính là số ảnh lật.
        f1 = macro_f1(pred["y_true"], pred["y_pred"], n_classes=2)
        acc = float((pred["y_true"] == pred["y_pred"]).mean())
        ref = json.loads(meta.read_text(encoding="utf-8"))["test"] if meta.exists() else {}
        ref_f1, ref_acc = ref.get("macro_f1"), ref.get("accuracy")
        d_f1 = abs(f1 - ref_f1) if ref_f1 is not None else 0.0
        d_acc = abs(acc - ref_acc) if ref_acc is not None else 0.0
        n_flipped = int(round(d_acc * len(pred["y_true"])))
        ok = n_flipped <= args.max_flipped
        ok_all = ok_all and ok

        print(f"  {tag}: {len(pred['y_true']):,} dự đoán -> {out.name} "
              f"({out.stat().st_size/1024:.0f} KB)")
        print(f"     Macro-F1 tính lại = {f1:.6f}  (đã báo cáo {ref_f1:.6f})  lệch {d_f1:.2e}")
        print(f"     Accuracy tính lại = {acc:.6f}  (đã báo cáo {ref_acc:.6f})  lệch {d_acc:.2e}")
        print(f"     -> {n_flipped} ảnh bị lật / {len(pred['y_true']):,}  "
              f"(dung sai cho phép {args.max_flipped} ảnh, do CPU vs {spec.get('device', 'GPU')})")
        if not ok:
            print("       !! lật quá nhiều ảnh — kiểm tra lại transform/kiến trúc khi nạp, "
                  "hoặc chạy lại trên đúng thiết bị đã huấn luyện.")

    print()
    print("KẾT LUẬN:", "tất cả khớp" if ok_all else "CÓ SAI LỆCH — xem ở trên")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
