import builtins
from pathlib import Path

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct, scalar
from torch import nn
from mmengine.logging import HistoryBuffer, MessageHub


class conv_block_nested(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        identity = x
        x = self.activation(self.bn1(x))
        x = self.bn2(self.conv2(x))
        return self.activation(x + identity)


class up(nn.Module):
    def __init__(self, in_ch: int, bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmod(avg_out + max_out)


class SNUNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, base_channel: int = 16) -> None:
        super().__init__()
        filters = [base_channel * 2**index for index in range(5)]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(
            filters[0] * 2 + filters[1], filters[0], filters[0]
        )
        self.conv1_1 = conv_block_nested(
            filters[1] * 2 + filters[2], filters[1], filters[1]
        )
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(
            filters[2] * 2 + filters[3], filters[2], filters[2]
        )
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(
            filters[3] * 2 + filters[4], filters[3], filters[3]
        )
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(
            filters[0] * 3 + filters[1], filters[0], filters[0]
        )
        self.conv1_2 = conv_block_nested(
            filters[1] * 3 + filters[2], filters[1], filters[1]
        )
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(
            filters[2] * 3 + filters[3], filters[2], filters[2]
        )
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(
            filters[0] * 4 + filters[1], filters[0], filters[0]
        )
        self.conv1_3 = conv_block_nested(
            filters[1] * 4 + filters[2], filters[1], filters[1]
        )
        self.Up1_3 = up(filters[1])
        self.conv0_4 = conv_block_nested(
            filters[0] * 5 + filters[1], filters[0], filters[0]
        )

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=4)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> tuple[torch.Tensor]:
        x0_0a = self.conv0_0(x_a)
        x1_0a = self.conv1_0(self.pool(x0_0a))
        x2_0a = self.conv2_0(self.pool(x1_0a))
        x3_0a = self.conv3_0(self.pool(x2_0a))

        x0_0b = self.conv0_0(x_b)
        x1_0b = self.conv1_0(self.pool(x0_0b))
        x2_0b = self.conv2_0(self.pool(x1_0b))
        x3_0b = self.conv3_0(self.pool(x2_0b))
        x4_0b = self.conv4_0(self.pool(x3_0b))

        x0_1 = self.conv0_1(torch.cat([x0_0a, x0_0b, self.Up1_0(x1_0b)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0a, x1_0b, self.Up2_0(x2_0b)], 1))
        x0_2 = self.conv0_2(
            torch.cat([x0_0a, x0_0b, x0_1, self.Up1_1(x1_1)], 1)
        )

        x2_1 = self.conv2_1(torch.cat([x2_0a, x2_0b, self.Up3_0(x3_0b)], 1))
        x1_2 = self.conv1_2(
            torch.cat([x1_0a, x1_0b, x1_1, self.Up2_1(x2_1)], 1)
        )
        x0_3 = self.conv0_3(
            torch.cat([x0_0a, x0_0b, x0_1, x0_2, self.Up1_2(x1_2)], 1)
        )

        x3_1 = self.conv3_1(torch.cat([x3_0a, x3_0b, self.Up4_0(x4_0b)], 1))
        x2_2 = self.conv2_2(
            torch.cat([x2_0a, x2_0b, x2_1, self.Up3_1(x3_1)], 1)
        )
        x1_3 = self.conv1_3(
            torch.cat([x1_0a, x1_0b, x1_1, x1_2, self.Up2_2(x2_2)], 1)
        )
        x0_4 = self.conv0_4(
            torch.cat([x0_0a, x0_0b, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1)
        )

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.stack([x0_1, x0_2, x0_3, x0_4]).sum(dim=0)
        out = self.ca(out) * (out + self.ca1(intra).repeat(1, 4, 1, 1))
        return (out,)


class SNUNetCD(nn.Module):
    def __init__(self, base_channel: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = SNUNetBackbone(in_channels=3, base_channel=base_channel)
        self.decode_head = nn.Module()
        self.decode_head.conv_seg = nn.Conv2d(
            base_channel * 4, num_classes, kernel_size=1
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x_a, x_b)
        return self.decode_head.conv_seg(features[-1])


def load_snunet_model(
    checkpoint_path: Path | str,
    device: torch.device | str,
) -> SNUNetCD:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SNUNet checkpoint not found: {checkpoint_path}")

    dtypes = (
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.bool_,
    )
    # Keep weights_only enabled while allowing the verified checkpoint's metadata types.
    safe_globals = [
        HistoryBuffer,
        MessageHub,
        _reconstruct,
        scalar,
        np.ndarray,
        np.dtype,
        builtins.getattr,
        *(type(np.dtype(dtype)) for dtype in dtypes),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError("Expected an Open-CD checkpoint containing a state_dict")

    model = SNUNetCD(base_channel=16, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()
