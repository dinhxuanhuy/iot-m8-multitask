"""Kiểm thử tự động cho thư viện M8.

Chạy::

    .\\.venv\\Scripts\\python.exe -m pytest tests -q

Các bài kiểm thử bám vào những tính chất *bắt buộc* của giao thức trong đề cương,
nên nếu có ai sửa mã làm sai giao thức thì kiểm thử sẽ thất bại.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8 import config as C  # noqa: E402
from m8.contrastive import nt_xent_loss  # noqa: E402
from m8.data import (  # noqa: E402
    RotationViewDataset,
    SuperclassDataset,
    _stratified_counts,
    audit_benchmark,
    build_benchmark,
    eval_transform,
    get_cifar10,
    labeled_train_transform,
    make_run_loaders,
    normalize_transform,
)
from m8.evaluate import ConfusionMatrix, evaluate_predictions, macro_f1  # noqa: E402
from m8.models import build_model, count_parameters, model_summary  # noqa: E402
from m8.train import make_optimizer, make_scheduler  # noqa: E402

EXPECTED_INFERENCE = 11_169_858
EXPECTED_TRAINING = 11_171_910


# ---------------------------------------------------------------- ánh xạ nhãn
def test_superclass_mapping_matches_outline():
    assert C.ANIMAL_CLASSES == (2, 3, 4, 5, 6, 7)
    assert C.VEHICLE_CLASSES == (0, 1, 8, 9)
    assert set(C.ANIMAL_CLASSES) | set(C.VEHICLE_CLASSES) == set(range(10))
    assert not (set(C.ANIMAL_CLASSES) & set(C.VEHICLE_CLASSES))
    # nhãn gốc 0..9 phải được ánh xạ đủ, không thiếu lớp nào
    assert len(C.ORIGINAL_TO_SUPER) == 10


def test_label_budget_matches_table_1():
    expected = {
        0.10: (4_500, 2_700, 1_800, 40_500),
        0.20: (9_000, 5_400, 3_600, 36_000),
        0.50: (22_500, 13_500, 9_000, 22_500),
    }
    for frac, (total, animal, vehicle, unlabeled) in expected.items():
        b = C.LABEL_BUDGET[frac]
        assert (b["total"], b["animal"], b["vehicle"], b["unlabeled"]) == (
            total, animal, vehicle, unlabeled
        )
        assert b["total"] + b["unlabeled"] == 45_000
        assert b["animal"] + b["vehicle"] == b["total"]
        # tỷ lệ Animal 60/40 phải được giữ trong ngân sách nhãn
        assert abs(b["animal"] / b["total"] - 0.60) < 1e-9


# ---------------------------------------------------------------- kiến trúc
def test_feature_shapes_match_table_4():
    model = build_model(with_rotation_head=True, seed=0)
    shapes = dict(model.encoder.feature_shapes())
    assert shapes["Conv 3x3 + BN + ReLU"] == (64, 32, 32)
    assert shapes["Nhóm residual 1: 2 BasicBlock"] == (64, 32, 32)
    assert shapes["Nhóm residual 2: 2 BasicBlock"] == (128, 16, 16)
    assert shapes["Nhóm residual 3: 2 BasicBlock"] == (256, 8, 8)
    assert shapes["Nhóm residual 4: 2 BasicBlock"] == (512, 4, 4)
    assert shapes["Global Average Pooling + Flatten"] == (512,)


def test_parameter_counts():
    model = build_model(with_rotation_head=True, seed=0)
    summary = model_summary(model)
    assert summary["encoder"] == 11_168_832
    assert summary["super_head"] == 1_026
    assert summary["rotation_head"] == 2_052
    assert summary["inference_total"] == EXPECTED_INFERENCE
    assert summary["training_total"] == EXPECTED_TRAINING
    # mô hình suy luận không tính Rotation Head
    assert summary["inference_total"] == summary["encoder"] + summary["super_head"]
    # tổng huấn luyện = tổng suy luận + Rotation Head
    assert summary["training_total"] == summary["inference_total"] + summary["rotation_head"]


def test_no_maxpool_and_stem_is_stride_one():
    model = build_model(with_rotation_head=False, seed=0)
    assert isinstance(model.encoder.conv1, torch.nn.Conv2d)
    assert model.encoder.conv1.kernel_size == (3, 3)
    assert model.encoder.conv1.stride == (1, 1)
    assert model.encoder.conv1.padding == (1, 1)
    assert model.encoder.conv1.bias is None
    # không được có MaxPool ở đầu mạng
    assert not any(isinstance(m, torch.nn.MaxPool2d) for m in model.encoder.modules())


def test_e1_has_no_rotation_head():
    e1 = build_model(with_rotation_head=False, seed=0)
    assert e1.rotation_head is None
    m8 = build_model(with_rotation_head=True, seed=0)
    assert m8.rotation_head is not None


def test_paired_init_shares_encoder_and_super_head():
    """Hai cấu hình ghép cặp phải dùng cùng trọng số khởi tạo của encoder và Superclass Head."""
    a = build_model(with_rotation_head=True, seed=42)
    state = {k: v.clone() for k, v in a.state_dict().items()}
    b = build_model(with_rotation_head=True, seed=999)   # seed khác -> trọng số khác
    assert not torch.allclose(a.super_head.weight, b.super_head.weight)

    from m8.models import load_shared_init

    load_shared_init(b, state)
    assert torch.allclose(a.encoder.conv1.weight, b.encoder.conv1.weight)
    assert torch.allclose(a.super_head.weight, b.super_head.weight)
    assert torch.allclose(a.rotation_head.weight, b.rotation_head.weight)


# ---------------------------------------------------------------- forward
def test_forward_shapes_and_joined_batch():
    model = build_model(with_rotation_head=True, seed=0)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 2)

    x_rot = torch.randn(4, 3, 32, 32)
    out_sup, out_rot = model.forward_joined(x, x_rot)
    assert out_sup.shape == (4, 2)
    assert out_rot.shape == (4, 4)

    # ghép batch: kết quả nhánh phân loại phải giống khi chạy riêng
    model.eval()
    with torch.no_grad():
        solo = model(x)
        joined, _ = model.forward_joined(x, x_rot)
    assert torch.allclose(solo, joined, atol=1e-5)


def test_rotation_head_learns_four_classes():
    model = build_model(with_rotation_head=True, seed=0)
    for k in range(4):
        x = torch.rot90(torch.randn(3, 3, 32, 32), k=k, dims=(2, 3))
        assert model.forward_rotation(x).shape == (3, 4)


# ---------------------------------------------------------------- rot90
def test_rot90_label_matches_transform():
    """Nhãn k do dataset sinh ra phải khớp đúng torch.rot90 của ảnh gốc."""
    train, _ = get_cifar10(download=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train), size=32, replace=False)

    from torchvision import transforms

    ds = RotationViewDataset(train, idx)
    to_tensor = transforms.ToTensor()
    for i in range(16):
        x, k, _ = ds[i]
        raw = to_tensor(train[int(idx[i])][0])
        norm = (raw - 0.5) / 0.5
        expect = torch.rot90(norm, k=k, dims=(1, 2))
        assert torch.allclose(expect, x, atol=1e-6), f"lệch tại mẫu {i}, k={k}"


def test_rotation_k_is_uniform_and_covers_all():
    train, _ = get_cifar10(download=False)
    ds = RotationViewDataset(train, np.arange(600))
    ks = np.array([ds[i][1] for i in range(600)])
    assert set(np.unique(ks)) == {0, 1, 2, 3}
    counts = np.bincount(ks, minlength=4)
    assert counts.min() > 100, counts


def test_normalize_constant_is_fixed():
    """Chuẩn hóa phải dùng hằng số 0,5 và không ước lượng từ dữ liệu."""
    assert C.NORMALIZE_MEAN == (0.5, 0.5, 0.5)
    assert C.NORMALIZE_STD == (0.5, 0.5, 0.5)
    # ảnh toàn 0 sau ToTensor phải thành -1; ảnh toàn 1 phải thành +1
    tf = normalize_transform()
    from PIL import Image

    black = tf(Image.new("RGB", (32, 32), (0, 0, 0)))
    white = tf(Image.new("RGB", (32, 32), (255, 255, 255)))
    assert torch.allclose(black, torch.full_like(black, -1.0), atol=1e-6)
    assert torch.allclose(white, torch.full_like(white, 1.0), atol=1e-6)


# ---------------------------------------------------------------- split
@pytest.fixture(scope="module")
def benchmark():
    import json

    cache = C.ARTIFACT_DIR / "splits.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        train, _ = get_cifar10(download=True)
        return build_benchmark(train.targets)
    train, _ = get_cifar10(download=True)
    return build_benchmark(train.targets)


def test_split_sizes_and_stratification(benchmark):
    assert benchmark.train_idx.size == 45_000
    assert benchmark.val_idx.size == 5_000
    assert benchmark.test_idx.size == 10_000
    assert len(np.intersect1d(benchmark.train_idx, benchmark.val_idx)) == 0


def test_label_subsets_are_nested_and_sized(benchmark):
    for seed in C.REPETITION_SEEDS:
        prev = None
        for frac in sorted(C.LABEL_FRACTIONS):
            dl = set(benchmark.dl(seed, frac).tolist())
            assert len(dl) == C.LABEL_BUDGET[frac]["total"]
            if prev is not None:
                assert prev.issubset(dl), f"tập {frac} không chứa tập nhỏ hơn (seed {seed})"
            prev = dl


def test_unlabeled_and_val_are_disjoint_from_labeled(benchmark):
    for seed in C.REPETITION_SEEDS:
        for frac in C.LABEL_FRACTIONS:
            dl = benchmark.dl(seed, frac)
            du = benchmark.du(seed, frac)
            assert len(np.intersect1d(dl, du)) == 0
            assert dl.size + du.size == 45_000
            assert len(np.intersect1d(np.concatenate([dl, du]), benchmark.val_idx)) == 0


def test_audit_report_flags_everything_ok(benchmark):
    train, _ = get_cifar10(download=False)
    audit = audit_benchmark(benchmark, train.targets)
    assert audit["no_overlap_train_val"] is True
    for info in audit["label_budget"].values():
        assert info["disjoint"] and info["no_val_leak"]
    for info in audit["nesting"].values():
        assert info["nested"]
    # hai siêu lớp có tỷ lệ 60/40 trên mọi tập
    assert abs(audit["train_animal_share"] - 0.60) < 1e-9
    assert abs(audit["val_animal_share"] - 0.60) < 1e-9


def test_stratified_counts_sums_exactly():
    labels = np.array([0] * 10 + [1] * 30 + [2] * 60)
    counts = _stratified_counts(labels, [0, 1, 2], 50)
    assert sum(counts.values()) == 50


# ---------------------------------------------------------------- loader
def test_e1_has_no_rotation_loader(benchmark):
    train, test = get_cifar10(download=False)
    loaders = make_run_loaders(train, test, benchmark, 42, 0.10, "none")
    assert "rotation_loader" not in loaders
    assert "rotation_ds" not in loaders


def test_e2l_rotation_source_is_labeled_only(benchmark):
    train, test = get_cifar10(download=False)
    loaders = make_run_loaders(train, test, benchmark, 42, 0.10, "labeled")
    assert len(loaders["rotation_ds"]) == C.LABEL_BUDGET[0.10]["total"]


def test_e2_rotation_source_is_full_train(benchmark):
    train, test = get_cifar10(download=False)
    loaders = make_run_loaders(train, test, benchmark, 42, 0.10, "train")
    assert len(loaders["rotation_ds"]) == 45_000


def test_labeled_dataset_returns_superclass_and_original(benchmark):
    train, _ = get_cifar10(download=False)
    ds = SuperclassDataset(train, benchmark.dl(42, 0.10), labeled_train_transform())
    x, y_super, y_orig = ds[0]
    assert x.shape == (3, 32, 32)
    assert C.ORIGINAL_TO_SUPER[int(y_orig)] == int(y_super)
    assert int(y_super) in (0, 1)


def test_rotation_dataset_has_no_superclass_labels():
    """Dataset nhánh phụ chỉ trả (ảnh, k, chỉ mục) — không thể rò rỉ nhãn siêu lớp."""
    train, _ = get_cifar10(download=False)
    ds = RotationViewDataset(train, np.arange(4))
    item = ds[0]
    assert len(item) == 3
    assert isinstance(item[1], int)
    assert not hasattr(ds, "super_labels")


# ---------------------------------------------------------------- chỉ số
def test_macro_f1_hand_computed_example():
    """Kiểm chứng Macro-F1 bằng ví dụ tính tay.

    y_true = [0,0,0,1,1,1], y_pred = [0,0,1,1,0,0]
      Ma trận nhầm lẫn: [[2,1],[2,1]]
      Animal  : TP=2, FP=2, FN=1 -> P=1/2, R=2/3, F1=4/7 ≈ 0,5714
      Vehicle : TP=1, FP=1, FN=2 -> P=1/2, R=1/3, F1=2/5  = 0,4
      Macro-F1 = (4/7 + 2/5)/2 ≈ 0,4857 ; Accuracy = 3/6 = 0,5
    """
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 0]
    rep = evaluate_predictions(y_true, y_pred)
    assert rep.accuracy == pytest.approx(0.5)
    assert rep.precision[0] == pytest.approx(0.5)
    assert rep.recall[0] == pytest.approx(2 / 3)
    assert rep.f1[0] == pytest.approx(4 / 7)
    assert rep.precision[1] == pytest.approx(0.5)
    assert rep.recall[1] == pytest.approx(1 / 3)
    assert rep.f1[1] == pytest.approx(0.4)
    assert rep.macro_f1 == pytest.approx((4 / 7 + 0.4) / 2)
    assert macro_f1(y_true, y_pred) == pytest.approx((4 / 7 + 0.4) / 2)
    assert rep.confusion == [[2, 1], [2, 1]]


def test_zero_division_convention_is_zero():
    cm = ConfusionMatrix(2)
    cm.update(np.zeros(10, dtype=int), np.zeros(10, dtype=int))   # không dự đoán lớp 1
    prec = cm.precision()
    assert prec[1] == 0.0
    rec = cm.recall()
    assert rec[1] == 0.0
    f1 = cm.f1()
    assert f1[1] == 0.0
    # Macro-F1 là trung bình hai lớp, không phải trung bình có trọng số
    assert abs(cm.macro_f1() - (f1[0] + f1[1]) / 2) < 1e-12


def test_all_animal_predictor_gets_60_percent_accuracy():
    """Bộ phân loại luôn đoán Animal đạt Accuracy 60 % trên phép chia 60/40."""
    y_true = np.array([0] * 6_000 + [1] * 4_000)
    y_pred = np.zeros_like(y_true)
    rep = evaluate_predictions(y_true, y_pred)
    assert abs(rep.accuracy - 0.60) < 1e-12
    assert rep.f1[1] == 0.0
    assert abs(rep.macro_f1 - rep.f1[0] / 2) < 1e-12


def test_report_serialization_roundtrip():
    rep = evaluate_predictions([0, 1, 1, 0], [0, 1, 0, 0])
    d = rep.as_dict()
    assert set(d) >= {"accuracy", "macro_f1", "precision", "recall", "f1", "confusion"}
    assert sum(sum(row) for row in d["confusion"]) == 4


# ---------------------------------------------------------------- huấn luyện
def test_optimizer_and_scheduler_config():
    model = build_model(with_rotation_head=True, seed=0)
    opt = make_optimizer(model)
    g = opt.param_groups[0]
    assert g["lr"] == C.LR
    assert g["weight_decay"] == C.WEIGHT_DECAY
    assert g["betas"] == C.BETAS
    assert g["eps"] == C.EPS
    sched = make_scheduler(opt, 10)
    # sau T_max bước, learning rate phải về eta_min = 0
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_gradient_flows_to_both_heads():
    model = build_model(with_rotation_head=True, seed=0)
    x_sup = torch.randn(4, 3, 32, 32)
    y_sup = torch.randint(0, 2, (4,))
    x_rot = torch.randn(4, 3, 32, 32)
    y_rot = torch.randint(0, 4, (4,))
    crit = torch.nn.CrossEntropyLoss()
    out_sup, out_rot = model.forward_joined(x_sup, x_rot)
    (crit(out_sup, y_sup) + 0.5 * crit(out_rot, y_rot)).backward()
    assert model.super_head.weight.grad is not None
    assert model.rotation_head.weight.grad is not None
    assert model.encoder.conv1.weight.grad is not None


def test_lambda_zero_still_changes_batchnorm_statistics():
    """Kiểm chứng lập luận ở mục 3.4: λ_rot = 0 KHÔNG làm nhánh phụ vô hại."""
    torch.manual_seed(0)
    model = build_model(with_rotation_head=True, seed=0).train()
    x_sup = torch.randn(8, 3, 32, 32)
    y_sup = torch.randint(0, 2, (8,))
    x_rot = torch.randn(8, 3, 32, 32) * 3.0     # phân bố khác rõ rệt
    y_rot = torch.randint(0, 4, (8,))
    crit = torch.nn.CrossEntropyLoss()
    opt = make_optimizer(model)

    def bn_state():
        return model.encoder.bn1.running_mean.clone(), model.encoder.bn1.running_var.clone()

    # Bước 1: chỉ nhánh phân loại
    before = bn_state()
    opt.zero_grad(set_to_none=True)
    crit(model(x_sup), y_sup).backward()
    opt.step()
    after_sup_only = bn_state()

    # Bước 2: λ_rot = 0 nhưng vẫn forward ảnh xoay -> running stats vẫn đổi khác
    opt.zero_grad(set_to_none=True)
    out_sup, out_rot = model.forward_joined(x_sup, x_rot)
    (crit(out_sup, y_sup) + 0.0 * crit(out_rot, y_rot)).backward()
    opt.step()
    after_with_rot = bn_state()

    assert not torch.allclose(after_sup_only[0], after_with_rot[0]), (
        "λ_rot = 0 vẫn phải làm thay đổi thống kê BatchNorm của baseline"
    )
    assert not torch.allclose(before[0], after_with_rot[0])


def test_seeds_are_reproducible():
    a = build_model(with_rotation_head=True, seed=7)
    b = build_model(with_rotation_head=True, seed=7)
    c = build_model(with_rotation_head=True, seed=8)
    assert count_parameters(a) == count_parameters(b)
    assert torch.allclose(a.encoder.conv1.weight, b.encoder.conv1.weight)
    assert not torch.allclose(a.encoder.conv1.weight, c.encoder.conv1.weight)


# ---------------------------------------------------------------- giao thức
def test_protocol_constants():
    assert C.EPOCHS == 100
    assert C.BATCH_SIZE == 128
    assert C.LAMBDA_ROT == 0.5
    assert C.REPETITION_SEEDS == (42, 43, 44)
    assert C.LABEL_FRACTIONS == (0.10, 0.20, 0.50)
    assert C.SPLIT_SEED == 2026
    assert C.MAIN_CONFIGS == ("E1", "E2-L", "E2")
    assert C.TEST_SUPER_COUNTS == (6_000, 4_000)
    assert C.VAL_SUPER_COUNTS == (3_000, 2_000)


def test_main_run_specs_cover_the_grid():
    specs = C.main_run_specs()
    assert len(specs) == 27
    combos = {(s.config, s.label_pct, s.seed) for s in specs}
    assert len(combos) == 27
    for spec in specs:
        assert spec.epochs == 100
        assert spec.lambda_rot == 0.5
    assert {s.rotation_source for s in specs} == {"none", "labeled", "train"}


def test_e1_spec_has_no_rotation_source():
    spec = C.RunSpec(config="E1", label_fraction=0.1, seed=42)
    assert spec.rotation_source == "none"


# ---------------------------------------------------------------- contrastive
def test_nt_xent_matches_manual_formula():
    """NT-Xent phải khớp công thức tính tay trên ví dụ chọn được."""
    from m8.contrastive import nt_xent_loss

    # view0/view2 cùng hướng e1 (positive của nhau), view1/view3 cùng hướng e2
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    tau = 0.5
    got = nt_xent_loss(z, temperature=tau).item()
    expect = math.log(math.exp(2.0) + 2.0) - 2.0
    assert abs(got - expect) < 1e-6


def test_nt_xent_degenerate_case_is_log_three():
    from m8.contrastive import nt_xent_loss

    z = torch.ones(4, 2)
    assert abs(nt_xent_loss(z, temperature=0.5).item() - math.log(3.0)) < 1e-6


def test_supcon_matches_manual_formula():
    from m8.contrastive import supcon_loss

    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    tau = 0.5
    got = supcon_loss(z, labels, temperature=tau).item()
    # tính tay theo công thức L_out
    zn = torch.nn.functional.normalize(z, dim=1)
    sim = zn @ zn.t() / tau
    manual = 0.0
    for i in range(4):
        others = [k for k in range(4) if k != i]
        lse = torch.logsumexp(sim[i, others], dim=0)
        pos = [k for k in others if labels[k] == labels[i]]
        manual += float((-torch.stack([sim[i, p] for p in pos]) + lse).mean())
    manual /= 4
    assert abs(got - manual) < 1e-5


def test_supcon_returns_finite_when_no_positive_pairs():
    """Trường hợp mỗi mẫu một lớp: không có positive -> 0, và KHÔNG được là NaN.

    Đây là hồi quy cho lỗi ``-inf * 0 = NaN`` phát hiện khi kiểm tra nhánh contrastive.
    """
    from m8.contrastive import supcon_loss

    z = torch.randn(6, 8)
    labels = torch.arange(6)
    out = supcon_loss(z, labels)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_supcon_is_finite_on_random_batch():
    from m8.contrastive import supcon_loss

    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(16, 32, generator=g)
        labels = torch.randint(0, 2, (16,), generator=g)
        assert torch.isfinite(supcon_loss(z, labels))


def test_projection_head_does_not_change_inference_cost():
    plain = build_model(with_rotation_head=True, seed=0)
    proj = build_model(with_rotation_head=True, proj_dim=64, seed=0)
    assert proj.inference_parameters() == plain.inference_parameters() == EXPECTED_INFERENCE
    assert proj.projection_parameters() > 0
    # cùng seed -> encoder và super head phải giống nhau
    assert torch.allclose(plain.encoder.conv1.weight, proj.encoder.conv1.weight)
    assert torch.allclose(plain.super_head.weight, proj.super_head.weight)


def test_projection_head_untrained_when_not_requested():
    model = build_model(with_rotation_head=True, seed=0)
    assert model.projection_head is None
    with pytest.raises(RuntimeError):
        model.forward_projection(torch.randn(2, 3, 32, 32))


def test_contrastive_dataset_returns_two_different_views(benchmark):
    from m8.data import ContrastiveViewDataset

    train, _ = get_cifar10(download=False)
    ds = ContrastiveViewDataset(train, benchmark.dl(42, 0.10), return_superclass=True)
    v1, v2, y, idx = ds[0]
    assert v1.shape == (3, 32, 32) and v2.shape == (3, 32, 32)
    assert int(y) in (0, 1)
    assert not torch.allclose(v1, v2), "hai view contrastive phải khác nhau"


def test_contrastive_dataset_can_hide_labels():
    """Nguồn D_train cho nhánh không giám sát phải KHÔNG mang nhãn (chống rò rỉ)."""
    from m8.data import ContrastiveViewDataset

    train, _ = get_cifar10(download=False)
    ds = ContrastiveViewDataset(train, np.arange(16), return_superclass=False)
    _, _, y, _ = ds[0]
    assert y == -1


def test_extension_configs_map_to_table_5():
    expect = {
        "E3": (False, "ntxent", "train"),
        "E4": (False, "supcon", "labeled"),
        "E5": (True, "ntxent", "train"),
        "E6": (True, "supcon", "labeled"),
    }
    for cfg, (rot, kind, source) in expect.items():
        spec = C.RunSpec(config=cfg, label_fraction=0.10, seed=42)
        assert spec.uses_rotation_head is rot, cfg
        assert spec.contrastive_kind == kind, cfg
        assert spec.contrastive_source == source, cfg


def test_main_configs_have_no_contrastive_branch():
    for cfg in C.MAIN_CONFIGS:
        spec = C.RunSpec(config=cfg, label_fraction=0.10, seed=42)
        assert spec.contrastive_kind is None
        assert spec.uses_projection_head is False


def test_ext_s1_loaders_have_no_rotation_and_have_contrastive(benchmark):
    from m8.data import make_run_loaders

    train, test = get_cifar10(download=False)
    for cfg in ("E3", "E4"):
        spec = C.RunSpec(config=cfg, label_fraction=0.10, seed=42)
        loaders = make_run_loaders(train, test, benchmark, spec.seed, spec.label_fraction,
                                   spec.rotation_source,
                                   contrastive_kind=spec.contrastive_kind,
                                   contrastive_source=spec.contrastive_source)
        assert "rotation_loader" not in loaders, cfg
        assert "contrastive_loader_main" in loaders, cfg
        expected = 45_000 if spec.contrastive_source == "train" else C.LABEL_BUDGET[0.10]["total"]
        assert len(loaders["contrastive_ds"]) == expected, cfg


def test_ext_loader_sources_match_protocol(benchmark):
    from m8.data import make_run_loaders

    train, test = get_cifar10(download=False)
    # E3/E5 dùng D_train; E4/E6 chỉ dùng D_L
    for cfg, n in (("E3", 45_000), ("E4", 4_500), ("E5", 45_000), ("E6", 4_500)):
        spec = C.RunSpec(config=cfg, label_fraction=0.10, seed=42)
        loaders = make_run_loaders(train, test, benchmark, spec.seed, spec.label_fraction,
                                   spec.rotation_source,
                                   contrastive_kind=spec.contrastive_kind,
                                   contrastive_source=spec.contrastive_source)
        assert len(loaders["contrastive_ds"]) == n, cfg


def test_extension_run_specs_grid():
    specs = C.extension_run_specs()
    assert len(specs) == 4 * len(C.EXTENSION_LABEL_FRACTIONS) * len(C.REPETITION_SEEDS)
    assert {s.config for s in specs} == set(C.EXTENSION_CONFIGS)
    for s in specs:
        assert s.epochs == 100
        assert s.lambda_c == C.LAMBDA_CONTRASTIVE


def test_contrastive_loss_weight_is_zero_when_lambda_zero():
    """Với λ_c = 0, gradient của nhánh contrastive phải bằng 0 (không đóng góp cập nhật)."""
    torch.manual_seed(0)
    model = build_model(with_rotation_head=False, proj_dim=32, seed=0).train()
    v1 = torch.randn(8, 3, 32, 32)
    v2 = torch.randn(8, 3, 32, 32)
    z = model.forward_projection(torch.cat([v1, v2], 0))
    # 0 * loss: giá trị đúng bằng 0
    loss = 0.0 * nt_xent_loss(z, temperature=0.2)
    assert loss.item() == 0.0
    loss.backward()
    # PyTorch vẫn tạo tensor grad (bằng 0) vì loss có phụ thuộc vào z, nhưng mọi phần tử
    # phải bằng 0 nên bước cập nhật của Projection Head là bước không.
    grad = dict(model.named_parameters())["projection_head.net.0.weight"].grad
    assert grad is not None
    assert torch.all(grad == 0)


def test_configured_temperatures_and_proj_dim_are_frozen():
    assert C.PROJ_DIM == 128
    assert C.TEMPERATURE_SUPCON == 0.1
    assert C.TEMPERATURE_NTXENT == 0.2
    assert C.LAMBDA_CONTRASTIVE > 0
    assert C.EXTENSION_LABEL_FRACTIONS == (0.10, 0.20)
