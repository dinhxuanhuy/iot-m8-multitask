"""Kiểm tra nhánh contrastive (E3--E6) trước khi chạy khối mở rộng.

Kiểm tra các tính chất bắt buộc của giao thức:

1. shape của hai view và nhãn trả về đúng;
2. giá trị hai hàm mất mát trên **ví dụ tính tay được**;
3. gradient chảy tới encoder qua Projection Head;
4. Projection Head không ảnh hưởng suy luận (số tham số suy luận không đổi);
5. dataset nguồn ``D_train`` cho nhánh không giám sát **không mang nhãn**, nên không
   thể rò rỉ nhãn thật của phần ẩn nhãn;
6. cấu hình E3--E6 ánh xạ đúng sang nhánh xoay và loại loss contrastive.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import (  # noqa: E402
    CONTRASTIVE_SOURCE,
    EXTENSION_CONFIGS,
    EXTENSION_SPEC,
    PROJ_DIM,
    TEMPERATURE_NTXENT,
    TEMPERATURE_SUPCON,
    RunSpec,
)
from m8.contrastive import nt_xent_loss, supcon_loss  # noqa: E402
from m8.data import ContrastiveViewDataset, build_benchmark, get_cifar10  # noqa: E402
from m8.models import build_model, model_summary  # noqa: E402
from m8.train import make_optimizer  # noqa: E402

EXPECTED_INFERENCE = 11_169_858


def section(title: str) -> None:
    print(f"\n== {title} ==", flush=True)


def main() -> int:
    ok = True

    # ---------------------------------------------------------------- 1. loss
    section("1. NT-Xent: ví dụ tính tay")
    # Chọn 4 vector sao cho positive có cosine 1 và hai negative có cosine 0:
    # view0/1 = e1, view2/3 = e2  -> positive của view0 (view2) có cosine 0.
    # Vì vậy dùng cặp (e1, e1) và (e2, e2) để tách bạch giá trị:
    # view0, view2 cùng hướng e1 (positive của nhau), view1, view3 cùng hướng e2.
    z = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    tau = 0.5
    loss = nt_xent_loss(z, temperature=tau)
    # view0: positive = view2 (cosine 1 -> logit 2), hai negative (cosine 0 -> logit 0)
    expected = math.log(math.exp(2.0) + 2.0) - 2.0
    print(f"  loss = {loss.item():.6f} | công thức tay = {expected:.6f}")
    if abs(loss.item() - expected) > 1e-6:
        print("  LỖI: NT-Xent không khớp công thức")
        ok = False

    # khi mọi view trùng nhau (mọi cosine = 1) mọi logit bằng nhau -> loss = log 3
    z_same = torch.ones(4, 2)
    loss_same = nt_xent_loss(z_same, temperature=tau)
    print(f"  loss khi mọi view trùng nhau = {loss_same.item():.6f} "
          f"(kỳ vọng log 3 = {math.log(3):.6f})")
    if abs(loss_same.item() - math.log(3)) > 1e-6:
        print("  LỖI: trường hợp suy biến không khớp log(3)")
        ok = False

    # loss phải bất biến khi hoán vị các view theo cùng một cặp positive
    perm = torch.tensor([2, 3, 0, 1])
    loss_perm = nt_xent_loss(z[perm], temperature=tau)
    print(f"  bất biến khi đổi thứ tự hai cặp: {abs(loss.item() - loss_perm.item()) < 1e-6}")
    if abs(loss.item() - loss_perm.item()) > 1e-6:
        print("  LỖI: NT-Xent không bất biến với hoán vị cặp")
        ok = False

    section("2. SupCon: ví dụ tính tay")
    # 4 view, hai lớp; view 0,1 cùng nhãn 0 (positives lẫn nhau), view 2,3 nhãn 1.
    z2 = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32)
    labels2 = torch.tensor([0, 0, 1, 1])
    tau2 = 0.5
    l_sup = supcon_loss(z2, labels2, temperature=tau2)
    # tính tay theo công thức L_out cho 4 view (mỗi view có đúng 1 positive)
    zn = F.normalize(z2, dim=1)
    sim = zn @ zn.t() / tau2
    manual = 0.0
    for i in range(4):
        others = [k for k in range(4) if k != i]
        lse = torch.logsumexp(sim[i, others], dim=0)
        pos = [k for k in others if labels2[k] == labels2[i]]
        manual += float((-torch.stack([sim[i, p] for p in pos]) + lse).mean())
    manual /= 4
    print(f"  loss = {l_sup.item():.6f} | tính tay = {manual:.6f}")
    if not torch.isfinite(l_sup):
        print("  LỖI: SupCon cho NaN/inf")
        ok = False
    elif abs(l_sup.item() - manual) > 1e-5:
        print("  LỖI: SupCon không khớp công thức")
        ok = False
    # SupCon phải khác NT-Xent vì dùng thêm thông tin nhãn
    l_nt = nt_xent_loss(torch.cat([z2, z2], dim=0), temperature=tau2)
    print(f"  (tham chiếu) NT-Xent trên cùng batch = {l_nt.item():.6f}")
    if not torch.isfinite(l_nt):
        print("  LỖI: NT-Xent không hữu hạn")
        ok = False
    # trường hợp mỗi mẫu là lớp riêng: không có positive -> trả 0 (không NaN)
    l_none = supcon_loss(z2, torch.arange(4), temperature=tau2)
    print(f"  không có positive pair -> loss = {l_none.item():.6f} (phải là 0)")
    if l_none.item() != 0.0:
        print("  LỖI: trường hợp không có positive phải trả 0")
        ok = False
    print(f"  nhiệt độ SupCon={TEMPERATURE_SUPCON}, NT-Xent={TEMPERATURE_NTXENT}")

    # ---------------------------------------------------------------- 3. mô hình
    section("3. Mô hình: Projection Head không ảnh hưởng suy luận")
    m_plain = build_model(with_rotation_head=True, seed=0)
    m_proj = build_model(with_rotation_head=True, proj_dim=PROJ_DIM, seed=0)
    s_plain, s_proj = model_summary(m_plain), model_summary(m_proj)
    print(f"  không có Projection Head: suy luận {s_plain['inference_total']:,} tham số")
    print(f"  có Projection Head      : suy luận {s_proj['inference_total']:,} tham số, "
          f"Projection Head {s_proj.get('projection_head', 0):,} tham số")
    if s_proj["inference_total"] != EXPECTED_INFERENCE:
        print("  LỖI: số tham số suy luận phải giữ nguyên", EXPECTED_INFERENCE)
        ok = False
    # encoder giống nhau vì cùng seed
    if not torch.allclose(m_plain.encoder.conv1.weight, m_proj.encoder.conv1.weight):
        print("  LỖI: cùng seed phải cho cùng trọng số encoder")
        ok = False

    section("4. Gradient qua Projection Head")
    model = build_model(with_rotation_head=True, proj_dim=PROJ_DIM, seed=0).train()
    v1 = torch.randn(8, 3, 32, 32)
    v2 = torch.randn(8, 3, 32, 32)
    z = model.forward_projection(torch.cat([v1, v2], dim=0))
    print(f"  shape vector projection = {tuple(z.shape)}")
    if z.shape != (16, PROJ_DIM):
        print("  LỖI: shape Projection Head sai")
        ok = False
    loss = nt_xent_loss(z, temperature=TEMPERATURE_NTXENT)
    loss.backward()
    for name in ("encoder.conv1.weight", "projection_head.net.0.weight", "super_head.weight"):
        p = dict(model.named_parameters())[name]
        got = p.grad is not None
        print(f"  grad {name:<34} = {got}")
    if model.encoder.conv1.weight.grad is None or \
       dict(model.named_parameters())["projection_head.net.0.weight"].grad is None:
        print("  LỖI: gradient không chảy qua nhánh contrastive")
        ok = False
    # Superclass Head KHÔNG nhận gradient từ loss contrastive
    if dict(model.named_parameters())["super_head.weight"].grad is not None:
        print("  LỖI: Superclass Head không được nhận gradient từ loss contrastive")
        ok = False

    # ---------------------------------------------------------------- 5. dữ liệu
    section("5. Dataset hai view")
    train, _ = get_cifar10(download=False)
    bench = build_benchmark(train.targets)
    dl_idx = bench.dl(42, 0.10)

    ds_sup = ContrastiveViewDataset(train, dl_idx, return_superclass=True)
    a, b, y, idx = ds_sup[0]
    print(f"  SupCon: view1 {tuple(a.shape)} | view2 {tuple(b.shape)} | nhãn {y} | idx {idx}")
    if a.shape != (3, 32, 32) or y not in (0, 1):
        print("  LỖI: shape/nhãn SupCon sai")
        ok = False
    if torch.allclose(a, b):
        print("  LỖI: hai view phải khác nhau (augmentation mạnh độc lập)")
        ok = False

    ds_unsup = ContrastiveViewDataset(train, bench.train_idx, return_superclass=False)
    _, _, y2, _ = ds_unsup[0]
    print(f"  NT-Xent với nguồn D_train: nhãn trả về = {y2} (phải là -1: không mang nhãn)")
    if y2 != -1:
        print("  LỖI: dataset không giám sát không được mang nhãn")
        ok = False

    # ---------------------------------------------------------------- 6. cấu hình
    section("6. Ánh xạ cấu hình E3--E6")
    for cfg in EXTENSION_CONFIGS:
        spec = RunSpec(config=cfg, label_fraction=0.10, seed=42)
        expect = EXTENSION_SPEC[cfg]
        print(f"  {cfg}: rotation_source={spec.rotation_source:<7} "
              f"contrastive={spec.contrastive_kind:<7} "
              f"nguồn={CONTRASTIVE_SOURCE[cfg]:<7} "
              f"rotation_head={spec.uses_rotation_head} "
              f"projection_head={spec.uses_projection_head}")
        if (spec.rotation_source != "none") != bool(expect["rotation"]):
            print("  LỖI: nhánh xoay không khớp bảng 5")
            ok = False
        if spec.contrastive_kind != expect["contrastive"]:
            print("  LỖI: loại loss contrastive không khớp bảng 5")
            ok = False

    print("\nKẾT LUẬN:", "ĐẠT" if ok else "KHÔNG ĐẠT", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
