"""Helpers for profiling activation distributions in sensitive layers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Sequence

import torch


def compute_activation_summary(tensor: torch.Tensor, percentiles: Sequence[float]) -> dict[str, float]:
    """Return summary statistics (max, percentile) for a tensor."""
    if tensor.numel() == 0:
        return {"max": 0.0, **{f"p{int(p*10000)}": 0.0 for p in percentiles}}

    flattened = tensor.detach().flatten()
    abs_values = flattened.abs().float()
    result = {"max": float(abs_values.max().item())}
    if percentiles:
        # torch.quantile on CUDA has a limit of ~16M elements.
        max_elems = 1_000_000
        if abs_values.numel() > max_elems:
            stride = abs_values.numel() // max_elems + 1
            abs_values = abs_values[::stride].contiguous()
        
        pct = torch.quantile(abs_values, torch.tensor(percentiles, dtype=abs_values.dtype, device=abs_values.device))
        for p, value in zip(percentiles, pct.reshape(-1)):
            key = f"p{int(p*10000)}"
            result[key] = float(value.item())
    return result


class ActivationStatsHook:
    """Forward hook that records percentile statistics for a module output (or input)."""

    def __init__(self, name: str, percentiles: Sequence[float], capture_input: bool = False) -> None:
        self.name = name
        self.percentiles = percentiles
        self.capture_input = capture_input
        self.records: list[dict[str, float]] = []
        self.per_channel_max: torch.Tensor | None = None

    def __call__(self, module, inputs, outputs) -> None:
        if self.capture_input:
            tensor = inputs[0]
        else:
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
        stats = compute_activation_summary(tensor, self.percentiles)
        stats["layer"] = self.name
        self.records.append(stats)
        
        # Update per-channel max (assuming tensor is [Batch, Seq, Channels])
        if tensor.dim() >= 2:
            # Flatten batch and seq dimensions
            current_max = tensor.abs().reshape(-1, tensor.shape[-1]).max(dim=0).values.detach().cpu()
            if self.per_channel_max is None:
                self.per_channel_max = current_max
            else:
                self.per_channel_max = torch.max(self.per_channel_max, current_max)

    def summary(self) -> dict[str, float]:
        if not self.records:
            return {}
        aggregated: dict[str, list[float]] = defaultdict(list)
        for record in self.records:
            for key, value in record.items():
                aggregated[key].append(value)

        return {key: float(sum(vals) / len(vals)) for key, vals in aggregated.items() if key != "layer"}


def sample_random_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape) * 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile activation statistics for sensitive layers")
    parser.add_argument("--percentiles", nargs="+", type=float, default=[0.999], help="Percentiles to capture")
    parser.add_argument("--shape", nargs=3, type=int, default=(4, 128, 4096), help="Synthetic tensor shape (B, T, D)")
    args = parser.parse_args()

    tensor = sample_random_tensor(tuple(args.shape))
    stats = compute_activation_summary(tensor, args.percentiles)
    print("Synthetic tensor activation stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()