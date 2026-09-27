"""Gói thư viện cho đề tài M8 — học đa nhiệm trên CIFAR-10."""

from . import (  # noqa: F401
    benchmark,
    config,
    contrastive,
    data,
    evaluate,
    experiments,
    models,
    train,
)

__all__ = [
    "config",
    "data",
    "models",
    "train",
    "contrastive",
    "evaluate",
    "benchmark",
    "experiments",
]
__version__ = "1.1.0"
