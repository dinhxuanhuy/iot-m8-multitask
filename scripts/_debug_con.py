"""In loss từng hàng để định vị sai số 2.0."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

sim_m = torch.tensor(
    [
        [-float("inf"), 2.0, 0.0, 0.0],
        [2.0, -float("inf"), 0.0, 0.0],
        [0.0, 0.0, -float("inf"), 2.0],
        [0.0, 0.0, 2.0, -float("inf")],
    ]
)
targets = torch.tensor([2, 3, 0, 1])

per_row = F.cross_entropy(sim_m, targets, reduction="none")
print("loss từng hàng:", [round(v, 6) for v in per_row.tolist()])
print("trung bình     :", F.cross_entropy(sim_m, targets).item())
print("giá trị tay mỗi hàng:", round(-2.0 + math.log(math.exp(2.0) + 2.0), 6))
print()
print("Hàng 0:", sim_m[0].tolist(), "target", int(targets[0]))
print("Hàng 2:", sim_m[2].tolist(), "target", int(targets[2]))
print()
# kiểm tra lại với target đúng là cặp positive của từng hàng
idx = torch.arange(4)
pos = (idx + 2) % 4
print("positive theo (i+N)%2N =", pos.tolist())
print("cross_entropy =", F.cross_entropy(sim_m, pos).item())
