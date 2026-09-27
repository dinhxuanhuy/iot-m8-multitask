"""Đo chi phí suy luận (mục 4.4).

Nguyên tắc:
  * đếm riêng tham số khi huấn luyện và khi suy luận;
  * đo dung lượng tệp chỉ chứa trọng số suy luận (không gộp optimizer state);
  * latency với batch = 1, tensor 1 × 3 × 32 × 32, FP32, ``model.eval()`` và
    ``torch.inference_mode()``;
  * khởi động 50 lượt rồi đo tối thiểu 1.000 lượt; nếu đo trên GPU thì đồng bộ
    CUDA trước và sau khối tính thời gian;
  * báo cáo mean / median / p95 theo ms/ảnh, lặp ba phiên và lưu log.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from .config import (
    LATENCY_ITERS,
    LATENCY_SESSIONS,
    LATENCY_WARMUP,
    set_global_threads,
)
from .models import MultiTaskNet, build_model, count_parameters

__all__ = [
    "environment_info",
    "parameter_report",
    "checkpoint_size_bytes",
    "measure_latency",
    "format_mib",
]


def format_mib(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def environment_info() -> Dict[str, object]:
    """Thông tin môi trường thực tế để ghi vào báo cáo."""
    import torchvision

    info: Dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "platform": platform.platform(),
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:
        import psutil  # type: ignore

        info["ram_gib"] = round(psutil.virtual_memory().total / 1024**3, 2)
        info["physical_cores"] = psutil.cpu_count(logical=False)
    except Exception:
        # Không có psutil: đọc RAM vật lý qua API Windows, không cần cài thêm gì.
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                info["ram_gib"] = round(stat.ullTotalPhys / 1024**3, 2)
        except Exception:
            pass
    return info


def parameter_report(model: MultiTaskNet) -> Dict[str, object]:
    """Số tham số huấn luyện / suy luận và dung lượng tệp trọng số suy luận."""
    import io

    enc = count_parameters(model.encoder)
    sup = count_parameters(model.super_head)
    rot = count_parameters(model.rotation_head) if model.rotation_head is not None else 0
    inference_state = {**model.encoder.state_dict(), **model.super_head.state_dict()}
    buf = io.BytesIO()
    torch.save(inference_state, buf)
    return {
        "encoder": enc,
        "super_head": sup,
        "rotation_head": rot,
        "training_total": enc + sup + rot,
        "inference_total": enc + sup,
        "inference_fp32_mib": format_mib((enc + sup) * 4),
        "inference_file_bytes": buf.getbuffer().nbytes,
        "inference_file_mib": format_mib(buf.getbuffer().nbytes),
    }


def checkpoint_size_bytes(model: MultiTaskNet, tmp_dir: Path | None = None) -> int:
    """Ghi riêng trọng số suy luận ra đĩa để lấy dung lượng tệp thực tế."""
    tmp_dir = Path(tmp_dir or ".")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "_inference_only.pt"
    torch.save({**model.encoder.state_dict(), **model.super_head.state_dict()}, path)
    size = path.stat().st_size
    path.unlink(missing_ok=True)
    return int(size)


@torch.inference_mode()
def measure_latency(
    model: MultiTaskNet,
    device: torch.device | str = "cpu",
    warmup: int = LATENCY_WARMUP,
    iters: int = LATENCY_ITERS,
    sessions: int = LATENCY_SESSIONS,
    threads: int | None = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Đo độ trễ suy luận batch = 1 và trả về thống kê theo ms/ảnh."""
    device = torch.device(device)
    n_threads = set_global_threads(threads)
    model = model.to(device).eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 3, 32, 32, generator=g).to(device)

    use_cuda = device.type == "cuda"

    def _sync() -> None:
        if use_cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        model(x)
    _sync()

    session_stats: List[Dict[str, float]] = []
    all_samples: List[float] = []
    for s in range(sessions):
        times = np.empty(iters, dtype=np.float64)
        for i in range(iters):
            _sync()
            t0 = time.perf_counter()
            model(x)
            _sync()
            times[i] = (time.perf_counter() - t0) * 1000.0
        all_samples.append(float(np.mean(times)))
        session_stats.append(
            {
                "session": s + 1,
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "std_ms": float(np.std(times, ddof=1)) if iters > 1 else 0.0,
            }
        )

    pooled = np.concatenate([np.asarray([st["mean_ms"]]) for st in session_stats])
    return {
        "device": str(device),
        "cpu_threads": n_threads,
        "batch_size": 1,
        "input_shape": [1, 3, 32, 32],
        "dtype": "float32",
        "warmup": warmup,
        "iters_per_session": iters,
        "sessions": sessions,
        "cuda_synchronized": use_cuda,
        "session_stats": session_stats,
        "mean_ms": float(np.mean(pooled)),
        "median_ms": float(statistics.median([st["median_ms"] for st in session_stats])),
        "p95_ms": float(np.mean([st["p95_ms"] for st in session_stats])),
        "session_mean_std_ms": float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0,
    }


def timing_breakdown_demo(
    device: torch.device | str = "cpu",
    n: int = 512,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Tách riêng thời gian đọc ảnh + tiền xử lý khỏi thời gian suy luận."""
    import time as _time

    from .data import eval_transform
    from .config import DATA_DIR
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(DATA_DIR), train=False, download=False)
    tf = eval_transform()
    t0 = _time.perf_counter()
    tensors = []
    for i in range(n):
        img, _ = ds[i]
        tensors.append(tf(img))
    pre_ms = (_time.perf_counter() - t0) * 1000.0
    return {
        "images": n,
        "read_and_preprocess_ms_per_image": pre_ms / n,
        "note": "Đọc ảnh + tiền xử lý CPU, đo riêng với thời gian suy luận.",
    }
