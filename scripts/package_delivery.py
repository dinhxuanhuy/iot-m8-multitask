"""Đóng gói gói bàn giao: checkpoint đã chọn bằng validation + dự đoán test + manifest.

Vì sao cần script này: đề cương §5.2 yêu cầu *"Đóng gói checkpoint được chọn bằng validation cho
các lượt chính; kèm logits hoặc dự đoán test theo chỉ mục để kiểm tra lại chỉ số."* Checkpoint
nặng ~46 MB mỗi lượt nên **không** được đưa thẳng vào kho mã nguồn; script này tạo một gói riêng
kèm manifest có băm SHA-256 để người nhận kiểm tra tính toàn vẹn.

Điểm quan trọng: script **kiểm chứng lại chỉ số** của từng checkpoint trước khi đóng gói. Một
checkpoint chỉ được coi là hợp lệ nếu nạp lại và tái lập đúng Macro-F1/Accuracy đã ghi trong
``<tag>.json`` (dung sai tính bằng số ảnh bị lật, vì CPU và GPU có thể lệch nhau vài ảnh).

Cách dùng::

    python scripts/package_delivery.py                       # chỉ lập manifest, không copy
    python scripts/package_delivery.py --ext                 # gồm cả khối mở rộng
    python scripts/package_delivery.py --zip                 # thực sự đóng gói thành .zip
    python scripts/package_delivery.py --out delivery_m8      # thư mục đích
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from m8.config import CHECKPOINT_DIR, CHECKPOINT_EXT_DIR, RESULTS_DIR  # noqa: E402

PRED_DIR = RESULTS_DIR / "test_predictions"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_checkpoint(ckpt: Path, meta: Path, max_flipped: int = 2) -> dict:
    """Nạp checkpoint, chạy lại trên D_test, đối chiếu chỉ số đã ghi.

    Trả về dict mô tả kết quả kiểm chứng. Không ném lỗi khi lệch — người gọi quyết định.
    """
    import torch

    from m8.data import SuperclassDataset, build_benchmark, eval_transform, get_cifar10
    from m8.evaluate import macro_f1
    from m8.models import build_model

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = blob["spec"]
    model = build_model(
        with_rotation_head=spec.get("rotation_source", "none") != "none",
        proj_dim=None,
    )
    missing, _ = model.load_state_dict(blob["state_dict"], strict=False)
    missing = [k for k in missing if "projection_head" not in k]
    if missing:
        return {"ok": False, "reason": f"thiếu khoá trọng số: {missing[:3]}"}

    model.eval()
    train, test = get_cifar10(download=False)
    bench = build_benchmark(train.targets)
    ds = SuperclassDataset(test, bench.test_idx, transform=eval_transform())
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            ps.append(model(x).argmax(dim=1).numpy())
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    f1 = macro_f1(y_true, y_pred, n_classes=2)
    acc = float((y_true == y_pred).mean())

    ref = json.loads(meta.read_text(encoding="utf-8"))["test"] if meta.exists() else {}
    ref_f1, ref_acc = ref.get("macro_f1"), ref.get("accuracy")
    n_flipped = int(round(abs(acc - ref_acc) * len(y_true))) if ref_acc is not None else -1
    ok = n_flipped >= 0 and n_flipped <= max_flipped
    return {
        "ok": ok,
        "macro_f1_recomputed": round(float(f1), 6),
        "macro_f1_recorded": None if ref_f1 is None else round(float(ref_f1), 6),
        "accuracy_recomputed": round(acc, 6),
        "accuracy_recorded": None if ref_acc is None else round(float(ref_acc), 6),
        "images_flipped": n_flipped,
    }


def collect(include_ext: bool) -> list[dict]:
    """Liệt kê các lượt cần đóng gói, kèm trạng thái từng tệp."""
    rows: list[dict] = []
    dirs = [(CHECKPOINT_DIR, "main")]
    if include_ext:
        dirs.append((CHECKPOINT_EXT_DIR, "extension"))
    for d, block in dirs:
        if not d.is_dir():
            continue
        for meta in sorted(d.glob("*.json")):
            tag = meta.stem
            ckpt = d / f"{tag}.pt"
            pred = PRED_DIR / f"{tag}.npz"
            rows.append({
                "block": block,
                "tag": tag,
                "meta": meta,
                "checkpoint": ckpt if ckpt.exists() else None,
                "predictions": pred if pred.exists() else None,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Đóng gói gói bàn giao")
    ap.add_argument("--out", default="delivery_m8", help="thư mục đích")
    ap.add_argument("--ext", action="store_true", help="gồm cả khối mở rộng")
    ap.add_argument("--zip", action="store_true", help="thực sự đóng gói thành .zip")
    ap.add_argument("--no-verify", action="store_true", help="bỏ bước kiểm chứng lại chỉ số")
    ap.add_argument("--max-flipped", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out)
    rows = collect(args.ext)
    if not rows:
        print(f"  !! không thấy lượt nào trong {CHECKPOINT_DIR}"
              + (f" hoặc {CHECKPOINT_EXT_DIR}" if args.ext else ""))
        print("     Trên máy đã huấn luyện, thư mục này chứa <tag>.json và <tag>.pt.")
        return 1

    print(f"  tìm thấy {len(rows)} lượt")
    manifest: list[dict] = []
    n_ck, n_pred, n_bad = 0, 0, 0

    for r in rows:
        entry: dict = {"block": r["block"], "tag": r["tag"]}
        meta = json.loads(r["meta"].read_text(encoding="utf-8"))
        spec = meta.get("spec", {})
        entry.update({
            "config": spec.get("config"),
            "label_pct": spec.get("label_pct"),
            "seed": spec.get("seed"),
            "epochs": spec.get("epochs"),
            "selected_epoch": meta.get("best_epoch"),
            "recorded_test_macro_f1": round(meta["test"]["macro_f1"], 6),
            "recorded_test_accuracy": round(meta["test"]["accuracy"], 6),
        })
        if r["checkpoint"]:
            entry["checkpoint_file"] = r["checkpoint"].name
            entry["checkpoint_bytes"] = r["checkpoint"].stat().st_size
            entry["checkpoint_sha256"] = sha256(r["checkpoint"])
            n_ck += 1
            if not args.no_verify:
                print(f"    kiểm chứng {r['tag']} …", flush=True)
                v = verify_checkpoint(r["checkpoint"], r["meta"], args.max_flipped)
                entry["verification"] = v
                if not v["ok"]:
                    n_bad += 1
                    print(f"      !! KHÔNG đạt: {v}")
                else:
                    print(f"      OK  lật {v['images_flipped']} ảnh, "
                          f"Macro-F1 {v['macro_f1_recomputed']}")
        else:
            entry["checkpoint_file"] = None
            entry["verification"] = {"ok": False, "reason": "không có tệp .pt"}

        if r["predictions"]:
            entry["predictions_file"] = r["predictions"].name
            entry["predictions_bytes"] = r["predictions"].stat().st_size
            entry["predictions_sha256"] = sha256(r["predictions"])
            n_pred += 1
        else:
            entry["predictions_file"] = None
        manifest.append(entry)

    # ---- ghi manifest ----
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "MANIFEST.json"
    payload = {
        "project": "M8 — Học đa nhiệm trên CIFAR-10",
        "note": ("Manifest gói bàn giao. Mỗi lượt ghi cấu hình, mức nhãn, seed, epoch được chọn "
                 "bằng validation, chỉ số đã báo cáo, và băm SHA-256 của checkpoint cùng dự đoán "
                 "test. Trường 'verification' ghi kết quả nạp lại checkpoint và tái lập chỉ số."),
        "n_runs": len(rows),
        "n_checkpoints": n_ck,
        "n_predictions": n_pred,
        "n_verification_failed": n_bad,
        "runs": manifest,
    }
    man_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  -> {man_path}")
    print(f"     {n_ck} checkpoint, {n_pred} tệp dự đoán, {n_bad} lượt kiểm chứng không đạt")

    if args.zip:
        zpath = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(man_path, man_path.name)
            for r in rows:
                if r["checkpoint"]:
                    z.write(r["checkpoint"], f"checkpoints/{r['checkpoint'].name}")
                if r["predictions"]:
                    z.write(r["predictions"], f"test_predictions/{r['predictions'].name}")
        print(f"  -> {zpath}  ({zpath.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  (chưa đóng gói — thêm --zip để tạo tệp .zip)")

    if n_ck == 0:
        print("\n  LƯU Ý: không có checkpoint nào trong thư mục này. Script đã lập manifest từ "
              "metadata,\n  nhưng cần chạy trên máy giữ results/checkpoints/*.pt để đóng gói đủ.")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
