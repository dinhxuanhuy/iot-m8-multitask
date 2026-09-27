"""Nhánh contrastive cho phần mở rộng E3--E6 (mục 3.3 của đề cương).

Hai mất mát được cài đặt trung thực theo tài liệu gốc và **không dùng nhãn thật của
phần ẩn nhãn**:

``L_NT-Xent`` (SimCLR, Chen và cộng sự 2020)
    Với minibatch gồm $2N$ view (hai view của $N$ ảnh), với mỗi view $i$ lấy view
    còn lại $j(i)$ của cùng ảnh làm positive, $2N-2$ view còn lại làm negative:

    .. math::
        \\ell_i = -\\log \\frac{\\exp(s_{i,j(i)}/\\tau)}
                       {\\sum_{k \\neq i} \\exp(s_{i,k}/\\tau)}, \\qquad
        L_{NT-Xent} = \\frac{1}{2N}\\sum_i \\ell_i

    $s$ là cosine similarity giữa các vector đã chuẩn hóa $L_2$. Không dùng nhãn siêu lớp.

``L_SupCon`` (Khosla và cộng sự 2020)
    Với $2N$ view và nhãn $\\tilde{y}$ của chúng, $A(i) = \\{a \\neq i : \\tilde{y}_a = \\tilde{y}_i\\}$:

    .. math::
        L_{SupCon} = \\sum_{i} \\frac{-1}{|A(i)|}
                     \\sum_{a \\in A(i)} \\log \\frac{\\exp(s_{i,a}/\\tau)}
                     {\\sum_{k \\neq i} \\exp(s_{i,k}/\\tau)}

    Positive pair được xây dựng từ **các view có cùng nhãn siêu lớp đã được cấp cho
    huấn luyện**; do đó trong đề tài này SupCon chỉ dùng $\\DL$.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ProjectionHead",
    "nt_xent_loss",
    "supcon_loss",
    "contrastive_loss",
]


class ProjectionHead(nn.Module):
    """Projection Head chỉ dùng khi huấn luyện (không tham gia suy luận).

    Kiến trúc: ``512 -> 512 -> proj_dim`` với ReLU ở lớp ẩn, theo SupCon/SimCLR.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pairwise_cosine(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Ma trận cosine similarity giữa các vector đã chuẩn hóa L2."""
    z = F.normalize(z, dim=1, eps=eps)
    return z @ z.t()


def nt_xent_loss(z: torch.Tensor, temperature: float = 0.2, eps: float = 1e-8) -> torch.Tensor:
    """NT-Xent cho minibatch ``z`` chứa $2N$ view: ``z[0:N]`` và ``z[N:2N]`` là hai view.

    Positive của view $i$ là $i+N$ (mod $2N$); mọi view khác đều là negative.
    """
    n2 = z.shape[0]
    if n2 % 2 != 0 or n2 < 4:
        raise ValueError("NT-Xent cần số view chẵn và >= 4")
    n = n2 // 2
    sim = _pairwise_cosine(z, eps) / temperature
    # loại bỏ phần tử trên đường chéo (view với chính nó)
    sim = sim.masked_fill(torch.eye(n2, dtype=torch.bool, device=z.device), float("-inf"))
    idx = torch.arange(n2, device=z.device)
    positive = (idx + n) % n2
    return F.cross_entropy(sim, positive)


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SupCon theo công thức ``L_out`` của Khosla và cộng sự (2020).

    ``z``: $2N$ vector đặc trưng (hai view của $N$ ảnh).
    ``labels``: nhãn siêu lớp của **cả hai** view, độ dài $2N$; ``labels[i+N] = labels[i]``.
    Mẫu nào không có positive trong batch sẽ được bỏ khỏi trung bình (tử số rỗng).
    """
    if z.shape[0] != labels.shape[0]:
        raise ValueError("số view và số nhãn phải bằng nhau")
    n2 = z.shape[0]
    device = z.device
    sim = _pairwise_cosine(z, eps) / temperature
    self_mask = torch.eye(n2, dtype=torch.bool, device=device)
    # mask positive: cùng nhãn và không phải chính nó
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask
    # log-sum-exp trên mọi mẫu khác (mẫu âm và mẫu dương)
    sim_masked = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)
    # QUAN TRỌNG: đặt log_prob trên đường chéo về 0 trước khi nhân với pos_mask.
    # Nếu giữ -inf tại đường chéo thì pos_mask (= 0 ở đó) nhân với -inf cho ra NaN,
    # và NaN đó lan ra toàn bộ loss.
    log_prob = torch.nan_to_num(log_prob, neginf=0.0)
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


def contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor | None,
    kind: str,
    temperature_supcon: float = 0.1,
    temperature_ntxent: float = 0.2,
) -> torch.Tensor:
    """Chọn mất mát contrastive theo ``kind`` (``"ntxent"`` hoặc ``"supcon"``)."""
    if kind == "ntxent":
        return nt_xent_loss(z, temperature_ntxent)
    if kind == "supcon":
        if labels is None:
            raise ValueError("SupCon cần nhãn siêu lớp")
        return supcon_loss(z, labels, temperature_supcon)
    raise ValueError(f"loại contrastive không hỗ trợ: {kind}")
