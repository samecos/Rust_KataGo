#!/usr/bin/env python3
"""Standalone 19x19 multihead Go models for the August self-play experiment.

Public contract::

    model = make_model("dense")  # or "compact"
    outputs = model(spatial, global_input)

``spatial`` is unpacked float32 [B, 22, 19, 19] and ``global_input`` is
float32 [B, 19]. Both variants return the same four *raw* outputs:

* policy_logits [B, 362] (row-major 361 board points, then pass)
* value_logits [B, 3] (win, loss, no result from the mover's perspective)
* score [B] (mover-perspective raw output; ``train_student.py`` uses 20-point units)
* ownership_logits [B, 361] (row-major mover-perspective ownership)

No softmax, sigmoid, legal-move mask, or engine-specific postprocessing is
applied. These deliberately small research models do not reproduce the full
TF3/KataGo inference-output contract and cannot be loaded by the engine.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


BOARD_SIZE = 19
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
SPATIAL_CHANNELS = 22
GLOBAL_CHANNELS = 19
POLICY_SIZE = BOARD_AREA + 1

# Keeping the same operators and heads isolates the effect of network size.
# Fewer/narrower convolutions give the compact arm a realistic MPS latency
# opportunity without relying on depthwise kernels whose latency is device
# dependent. A speed gain must still be measured on saved checkpoints.
MODEL_SPECS: dict[str, tuple[int, int]] = {
    "dense": (64, 6),
    "compact": (48, 4),
}


class StudentOutputs(NamedTuple):
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    score: torch.Tensor
    ownership_logits: torch.Tensor


class ResidualBlock(nn.Module):
    """Two padded 3x3 convolutions with an identity skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.conv1(x))
        residual = self.conv2(residual)
        return F.relu(x + residual)


class StudentNet(nn.Module):
    """Compact spatial trunk with full-board policy and auxiliary outputs."""

    def __init__(self, *, variant: str, width: int, blocks: int) -> None:
        super().__init__()
        if width <= 0 or blocks <= 0:
            raise ValueError("width and blocks must be positive")
        self.variant = variant
        self.width = width
        self.blocks_count = blocks

        self.stem = nn.Conv2d(SPATIAL_CHANNELS, width, kernel_size=3, padding=1)
        self.global_to_stem = nn.Linear(GLOBAL_CHANNELS, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))

        # A pointwise map preserves board coordinates. Pass is predicted from
        # the pooled board state and the same 19 global inputs.
        self.policy_board = nn.Conv2d(width, 1, kernel_size=1)
        self.policy_pass = nn.Linear(width + GLOBAL_CHANNELS, 1)

        self.value_features = nn.Linear(width + GLOBAL_CHANNELS, 64)
        self.value = nn.Linear(64, 3)
        self.score_head = nn.Linear(64, 1)
        self.ownership = nn.Conv2d(width, 1, kernel_size=1)

    def forward(
        self, spatial: torch.Tensor, global_input: torch.Tensor
    ) -> StudentOutputs:
        if spatial.ndim != 4 or tuple(spatial.shape[1:]) != (
            SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE
        ):
            raise ValueError("spatial must have shape [B,22,19,19]")
        if global_input.ndim != 2 or global_input.shape != (
            spatial.shape[0], GLOBAL_CHANNELS
        ):
            raise ValueError("global_input must have shape [B,19] with matching B")

        stem_global = self.global_to_stem(global_input)[:, :, None, None]
        x = self.blocks(F.relu(self.stem(spatial) + stem_global))
        pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        combined = torch.cat((pooled, global_input), dim=1)

        board_logits = self.policy_board(x).flatten(1)
        pass_logit = self.policy_pass(combined)
        policy_logits = torch.cat((board_logits, pass_logit), dim=1)

        value_features = F.relu(self.value_features(combined))
        return StudentOutputs(
            policy_logits=policy_logits,
            value_logits=self.value(value_features),
            score=self.score_head(value_features).squeeze(1),
            ownership_logits=self.ownership(x).flatten(1),
        )


def make_model(variant: str) -> StudentNet:
    """Construct one of the frozen comparison architectures."""
    try:
        width, blocks = MODEL_SPECS[variant]
    except KeyError as exc:
        raise ValueError(
            f"unknown model variant {variant!r}; choose {', '.join(MODEL_SPECS)}"
        ) from exc
    return StudentNet(variant=variant, width=width, blocks=blocks)


def count_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print standalone student model sizes")
    parser.add_argument("--variant", choices=tuple(MODEL_SPECS), help="print only one variant")
    args = parser.parse_args()
    for variant in ((args.variant,) if args.variant else MODEL_SPECS):
        model = make_model(variant)
        print(
            f"{variant}: width={model.width}, residual_blocks={model.blocks_count}, "
            f"parameters={count_parameters(model):,}"
        )


if __name__ == "__main__":
    main()
