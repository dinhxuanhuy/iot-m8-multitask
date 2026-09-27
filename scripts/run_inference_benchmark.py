"""Đo chi phí suy luận và ghi môi trường thực nghiệm vào ``results/``.

Sinh ra:
  * ``results/environment.json``          — thông tin môi trường thật;
  * ``results/inference_benchmark.json``  — số tham số, dung lượng, độ trễ CPU và GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from m8.benchmark import (  # noqa: E402
    checkpoint_size_bytes,
    environment_info,
    measure_latency,
    parameter_report,
    timing_breakdown_demo,
)
from m8.config import LATENCY_ITERS, LATENCY_SESSIONS, LATENCY_WARMUP, RESULTS_DIR  # noqa: E402
from m8.models import build_model  # noqa: E402


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    print("== Môi trường ==", flush=True)
    env = environment_info()
    (RESULTS_DIR / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(env, ensure_ascii=False, indent=2), flush=True)
    out["environment"] = env

    print("\n== Số tham số và dung lượng ==", flush=True)
    model = build_model(with_rotation_head=True, seed=2026, device="cpu")
    params = parameter_report(model)
    params["inference_file_bytes_measured"] = checkpoint_size_bytes(model, RESULTS_DIR)
    params["inference_file_mib_measured"] = round(
        params["inference_file_bytes_measured"] / 1024**2, 4
    )
    print(json.dumps(params, ensure_ascii=False, indent=2), flush=True)
    out["parameters"] = params

    print("\n== Độ trễ CPU (4 luồng) ==", flush=True)
    cpu = measure_latency(model, device="cpu", warmup=LATENCY_WARMUP, iters=LATENCY_ITERS,
                          sessions=LATENCY_SESSIONS, threads=4)
    print(json.dumps({k: v for k, v in cpu.items() if k != "session_stats"},
                     ensure_ascii=False, indent=2), flush=True)
    out["latency_cpu"] = cpu

    if torch.cuda.is_available():
        print("\n== Độ trễ GPU ==", flush=True)
        torch.cuda.empty_cache()
        gpu = measure_latency(model, device="cuda", warmup=LATENCY_WARMUP, iters=LATENCY_ITERS,
                              sessions=LATENCY_SESSIONS, threads=4)
        print(json.dumps({k: v for k, v in gpu.items() if k != "session_stats"},
                         ensure_ascii=False, indent=2), flush=True)
        out["latency_gpu"] = gpu
    else:
        out["latency_gpu"] = {}

    print("\n== Tách riêng thời gian đọc ảnh + tiền xử lý ==", flush=True)
    pre = timing_breakdown_demo(n=512)
    print(json.dumps(pre, ensure_ascii=False, indent=2), flush=True)
    out["preprocess_ms_per_image"] = pre["read_and_preprocess_ms_per_image"]

    path = RESULTS_DIR / "inference_benchmark.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
