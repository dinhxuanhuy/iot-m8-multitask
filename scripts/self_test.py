"""Kiểm tra kỹ thuật tự động cho mã nguồn M8 (tuần 1 + phụ lục đề cương).

Chạy script này để xác nhận:
  * chuỗi kích thước tensor của stem/layer khớp bảng 4;
  * số tham số suy luận = 11.169.858 và số tham số khi thêm Rotation Head = 11.171.910;
  * một lượt forward/backward cho từng cấu hình E1, E2-L, E2 chạy được;
  * phép chia benchmark không trùng lặp và các tập nhãn lồng nhau đúng.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.benchmark import environment_info, parameter_report  # noqa: E402
from m8.config import ARTIFACT_DIR, RESULTS_DIR  # noqa: E402
from m8.data import (  # noqa: E402
    RotationViewDataset,
    SuperclassDataset,
    audit_benchmark,
    build_benchmark,
    get_cifar10,
    labeled_train_transform,
)
from m8.models import build_model, count_parameters  # noqa: E402
from m8.train import make_optimizer  # noqa: E402

EXPECTED_INFERENCE = 11_169_858
EXPECTED_TRAINING = 11_171_910


def main() -> int:
    ok = True
    print("== Môi trường ==", flush=True)
    env = environment_info()
    print(json.dumps(env, ensure_ascii=False, indent=2), flush=True)

    print("\n== Chuỗi tensor (bảng 4) ==", flush=True)
    model = build_model(with_rotation_head=True, seed=1234)
    for name, shape in model.encoder.feature_shapes():
        print(f"  {name:<44} {shape}", flush=True)

    print("\n== Số tham số ==", flush=True)
    r = parameter_report(model)
    for k, v in r.items():
        print(f"  {k:<26} {v}", flush=True)
    if r["inference_total"] != EXPECTED_INFERENCE:
        print(f"  LỆCH: kỳ vọng {EXPECTED_INFERENCE}, nhận {r['inference_total']}", flush=True)
        ok = False
    if r["training_total"] != EXPECTED_TRAINING:
        print(f"  LỆCH: kỳ vọng {EXPECTED_TRAINING}, nhận {r['training_total']}", flush=True)
        ok = False

    print("\n== Forward/backward từng cấu hình ==", flush=True)
    x_sup = torch.randn(8, 3, 32, 32)
    y_sup = torch.randint(0, 2, (8,))
    x_rot = torch.randn(8, 3, 32, 32)
    y_rot = torch.randint(0, 4, (8,))
    crit = torch.nn.CrossEntropyLoss()

    # E1: chỉ nhánh phân loại
    m1 = build_model(with_rotation_head=False, seed=42)
    loss = crit(m1(x_sup), y_sup)
    loss.backward()
    print(f"  E1   loss={loss.item():.4f}  inference_params={m1.inference_parameters()}", flush=True)

    # E2-L / E2: hai nhánh ghép batch
    m2 = build_model(with_rotation_head=True, seed=42)
    out_sup, out_rot = m2.forward_joined(x_sup, x_rot)
    loss = crit(out_sup, y_sup) + 0.5 * crit(out_rot, y_rot)
    loss.backward()
    print(f"  E2-L loss={loss.item():.4f}  inference_params={m2.inference_parameters()}", flush=True)
    print(f"  Rotation Head chạy đủ bốn góc:", flush=True)
    for k in range(4):
        logits = m2.forward_rotation(torch.rot90(x_rot, k=k, dims=(2, 3)))
        print(f"    k={k} -> logits {tuple(logits.shape)}", flush=True)

    print("\n== Kiểm tra optimizer/scheduler ==", flush=True)
    opt = make_optimizer(m2)
    print(f"  AdamW groups={len(opt.param_groups)} lr={opt.param_groups[0]['lr']} "
          f"wd={opt.param_groups[0]['weight_decay']}", flush=True)

    print("\n== Kiểm tra phép chia benchmark ==", flush=True)
    train, _ = get_cifar10(download=True)
    bench = build_benchmark(train.targets)
    audit = audit_benchmark(bench, train.targets)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    for info in audit["label_budget"].values():
        ok = ok and info["disjoint"] and info["no_val_leak"]
    for info in audit["nesting"].values():
        ok = ok and info["nested"]
    ok = ok and audit["no_overlap_train_val"]

    print("\n== Kiểm tra dataset xoay ==", flush=True)
    idx = bench.dl(42, 0.10)[:64]
    rot_ds = RotationViewDataset(train, idx)
    xs, ks, ids = [], [], []
    for i in range(64):
        x, k, j = rot_ds[i]
        xs.append(x)
        ks.append(k)
        ids.append(j)
    X = torch.stack(xs)
    K = torch.tensor(ks)
    print(f"  batch shape={tuple(X.shape)}  phân bố k={torch.bincount(K, minlength=4).tolist()}", flush=True)
    # kiểm tra tensor xoay khớp torch.rot90 của ảnh gốc
    lab_ds = SuperclassDataset(train, idx, labeled_train_transform())
    ok_rot = True
    for i in range(8):
        x_rot_i = X[i]
        k = int(K[i])
        # dựng lại ảnh chuẩn hóa không augmentation từ chính ảnh gốc
        from m8.data import normalize_transform
        img, _, _ = lab_ds[i]
        raw = torch.tensor(np.asarray(train[int(ids[i])][0]), dtype=torch.float32).permute(2, 0, 1) / 255.0
        norm = (raw - 0.5) / 0.5
        expect = torch.rot90(norm, k=k, dims=(1, 2))
        if not torch.allclose(expect, x_rot_i, atol=1e-5):
            ok_rot = False
            print(f"  LỆCH tại mẫu {i}, k={k}", flush=True)
    print(f"  kiểm tra rot90 khớp: {ok_rot}", flush=True)
    ok = ok and ok_rot

    # lưu báo cáo
    out = RESULTS_DIR / "self_test.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "environment": env,
                "parameters": {k: (float(v) if isinstance(v, float) else v) for k, v in r.items()},
                "feature_shapes": [[n, list(s)] for n, s in model.encoder.feature_shapes()],
                "benchmark_audit": audit,
                "rotation_check_ok": ok_rot,
                "passed": bool(ok),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nĐã lưu {out}", flush=True)
    print("KẾT LUẬN:", "ĐẠT" if ok else "KHÔNG ĐẠT", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
