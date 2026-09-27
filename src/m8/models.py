"""Kiến trúc mô hình cho đề tài M8 (mục 3.1, bảng 4).

ResNet-18 điều chỉnh cho ảnh 32 × 32:
  * không dùng trọng số tiền huấn luyện (khởi tạo ngẫu nhiên);
  * stem ``Conv2d(3, 64, 3, stride=1, padding=1, bias=False)``;
  * bỏ ``MaxPool`` đầu mạng;
  * bốn nhóm residual, mỗi nhóm hai BasicBlock;
  * Global Average Pooling -> vector 512 chiều;
  * hai head tuyến tính có bias: Superclass 512->2, Rotation 512->4.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

__all__ = [
    "ResNet18CIFAR",
    "MultiTaskNet",
    "build_model",
    "count_parameters",
]


class ResNet18CIFAR(nn.Module):
    """ResNet-18 cho ảnh 32 × 32, không có sẵn classification head."""

    def __init__(self, widths: Tuple[int, int, int, int] = (64, 128, 256, 512)) -> None:
        super().__init__()
        self.inplanes = widths[0]
        self.widths = widths

        # stem: 3x3 conv, stride 1, không maxpool
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(widths[0], blocks=2, stride=1)
        self.layer2 = self._make_layer(widths[1], blocks=2, stride=2)
        self.layer3 = self._make_layer(widths[2], blocks=2, stride=2)
        self.layer4 = self._make_layer(widths[3], blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_features = widths[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers: List[nn.Module] = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def feature_shapes(self, device: torch.device | None = None) -> List[Tuple[str, Tuple[int, ...]]]:
        """Chuỗi kích thước tensor để đưa vào bảng 4 của báo cáo."""
        device = device or torch.device("cpu")
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 3, 32, 32, device=device)
        shapes: List[Tuple[str, Tuple[int, ...]]] = [("Ảnh đầu vào", tuple(x.shape[1:]))]
        with torch.no_grad():
            x = self.relu(self.bn1(self.conv1(x)))
            shapes.append(("Conv 3x3 + BN + ReLU", tuple(x.shape[1:])))
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=1):
                x = layer(x)
                shapes.append((f"Nhóm residual {i}: 2 BasicBlock", tuple(x.shape[1:])))
            x = self.avgpool(x)
            shapes.append(("Global Average Pooling + Flatten", (int(x.flatten(1).shape[1]),)))
        if was_training:
            self.train()
        return shapes


class MultiTaskNet(nn.Module):
    """Encoder ResNet-18 dùng chung + head siêu lớp + head xoay.

    Cả hai nhánh nhận **view khác nhau** và ghép theo chiều batch trước khi qua
    encoder; các lớp BatchNorm được dùng chung nên thống kê chạy của mạng phụ
    thuộc vào việc có nhánh xoay hay không (điều này được nêu rõ trong mục 3.4).
    """

    def __init__(
        self,
        num_super: int = 2,
        num_rotations: int = 4,
        with_rotation_head: bool = True,
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ResNet18CIFAR()
        feat = self.encoder.out_features
        self.super_head = nn.Linear(feat, num_super, bias=True)
        self.rotation_head = nn.Linear(feat, num_rotations, bias=True) if with_rotation_head else None
        self.with_rotation_head = with_rotation_head
        # Projection Head chỉ dùng khi huấn luyện nhánh contrastive (E3--E6)
        self.proj_dim = proj_dim
        if proj_dim is not None:
            from .contrastive import ProjectionHead

            self.projection_head: nn.Module | None = ProjectionHead(feat, feat, proj_dim)
        else:
            self.projection_head = None

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.super_head(self.encoder(x))

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Đặc trưng encoder -> Projection Head (chỉ dùng khi huấn luyện)."""
        if self.projection_head is None:
            raise RuntimeError("mô hình không có Projection Head")
        return self.projection_head(self.encoder(x))

    def forward_joined(self, x_sup: torch.Tensor, x_rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ghép hai minibatch theo chiều batch rồi tách đặc trưng về đúng head."""
        assert self.rotation_head is not None
        b = x_sup.shape[0]
        joined = torch.cat([x_sup, x_rot], dim=0)
        feats = self.encoder(joined)
        f_sup, f_rot = feats[:b], feats[b:]
        return self.super_head(f_sup), self.rotation_head(f_rot)

    def forward_rotation(self, x_rot: torch.Tensor) -> torch.Tensor:
        assert self.rotation_head is not None
        return self.rotation_head(self.encoder(x_rot))

    def inference_parameters(self) -> int:
        """Số tham số khi suy luận: encoder + Superclass Head (mục 4.4)."""
        return count_parameters(self.encoder) + count_parameters(self.super_head)

    def training_parameters(self) -> int:
        names = ["encoder", "super_head"] + (["rotation_head"] if self.with_rotation_head else [])
        return sum(count_parameters(getattr(self, n)) for n in names)

    def projection_parameters(self) -> int:
        return count_parameters(self.projection_head) if self.projection_head is not None else 0


def count_parameters(module: nn.Module) -> int:
    """Đếm tham số, kể cả buffer? Không: chỉ ``nn.Parameter`` (trọng số học được)."""
    return int(sum(p.numel() for p in module.parameters()))


def build_model(
    num_super: int = 2,
    num_rotations: int = 4,
    with_rotation_head: bool = True,
    proj_dim: int | None = None,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> MultiTaskNet:
    """Khởi tạo ngẫu nhiên (không dùng trọng số tiền huấn luyện).

    Nếu ``seed`` được cấp, việc khởi tạo là xác định để các cấu hình ghép cặp
    dùng **cùng trọng số khởi tạo của encoder và Superclass Head** (bảng 7).
    """
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        model = MultiTaskNet(num_super=num_super, num_rotations=num_rotations,
                             with_rotation_head=with_rotation_head, proj_dim=proj_dim)
    return model.to(device)


def shared_init_state(model: MultiTaskNet) -> Dict[str, torch.Tensor]:
    """Trạng thái khởi tạo dùng chung cho một cặp so sánh ghép cặp."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_shared_init(model: MultiTaskNet, state: Dict[str, torch.Tensor]) -> None:
    """Nạp lại trọng số khởi tạo dùng chung; chỉ những khóa cùng tên mới được nạp.

    Trọng số của Rotation Head không nằm trong ``state`` của E1 (không có head phụ),
    nên head này vẫn giữ giá trị khởi tạo riêng của nó.
    """
    own = model.state_dict()
    patch = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(patch)
    model.load_state_dict(own)


def model_summary(model: MultiTaskNet) -> Dict[str, int]:
    """Số tham số từng phần (bảng báo cáo mục 4.4)."""
    out = {
        "encoder": count_parameters(model.encoder),
        "super_head": count_parameters(model.super_head),
    }
    if model.rotation_head is not None:
        out["rotation_head"] = count_parameters(model.rotation_head)
        out["training_total"] = model.training_parameters()
        out["inference_total"] = model.inference_parameters()
    else:
        out["training_total"] = out["encoder"] + out["super_head"]
        out["inference_total"] = out["training_total"]
    if model.projection_head is not None:
        out["projection_head"] = model.projection_parameters()
        out["training_total_with_projection"] = out["training_total"] + out["projection_head"]
    return out
