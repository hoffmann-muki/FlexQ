"""Pluggable clipping module for OACS tuning.

This module provides functionality to build and apply adaptive clipping schedules
based on layer activation statistics. It can be enabled/disabled in the main tuning workflow.

Usage:
    from algorithm.clipping_module import ClippingModule
    clipper = ClippingModule()
    schedule = clipper.build_schedule(profiles, clip_pct, target_key, bonus_scale, bonus_cap, zero_shift)
"""

import math
from pathlib import Path
import csv
import time
from typing import Sequence, Dict


class ClippingModule:
    """Handles adaptive clipping for OACS (Outlier-Aware Clipping and Smoothing)."""

    def __init__(self):
        pass

    def build_layer_clip_schedule(
        self,
        stats: Dict[int, Dict[str, float]],
        base_clip_pct: float,
        target_percentile_key: str,
        bonus_scale: float,
        bonus_cap: float,
        zero_shift: float,
    ) -> Dict[int, Dict[str, float]]:
        """Build a per-layer clipping schedule based on activation stats.

        Args:
            stats: Layer stats from profiling.
            base_clip_pct: Base clipping percentile (e.g., 0.995).
            target_percentile_key: Key for target percentile (e.g., 'p9000').
            bonus_scale: Scale for bonus clipping based on severity.
            bonus_cap: Max bonus clipping fraction.
            zero_shift: Zero-point shift scale.

        Returns:
            Schedule dict: layer_idx -> {'clip_percentile', 'zero_shift_scale', 'severity'}
        """
        schedule: Dict[int, Dict[str, float]] = {}
        for layer_idx, layer_stats in stats.items():
            target_value = layer_stats.get(target_percentile_key, layer_stats.get("max", 0.0))
            percentile_keys = [k for k in layer_stats.keys() if k.startswith("p")]
            severity_key = max(percentile_keys, key=lambda k: float(k.lstrip("p") or 0), default=target_percentile_key)
            severity_value = layer_stats.get(severity_key, target_value)
            if target_value > 0:
                severity = (severity_value + 1e-9) / (target_value + 1e-9)
            else:
                severity = 1.0

            # Use log scale for severity to handle extreme outliers without saturating immediately
            log_severity = math.log(max(1.0, severity))
            bonus = min(max(0.0, log_severity * bonus_scale), bonus_cap)

            # Cap at 0.9999 ONLY if we are actually trying to clip (base < 1.0).
            # If base is 1.0 (baseline), we allow 1.0 to pass through.
            if base_clip_pct >= 1.0:
                adjusted_clip = 1.0
            else:
                adjusted_clip = min(0.9999, base_clip_pct + bonus)

            schedule[layer_idx] = {
                "clip_percentile": adjusted_clip,
                "zero_shift_scale": zero_shift,
                "severity": severity,
            }
        return schedule

    def log_layer_schedule(
        self,
        schedule: Dict[int, Dict[str, float]],
        clip_pct: float,
        zero_shift: float,
        path: str | None,
        target_key: str,
    ) -> None:
        """Log the layer clipping schedule to CSV.

        Args:
            schedule: The built schedule.
            clip_pct: Base clip percentile.
            zero_shift: Zero shift scale.
            path: CSV path to append to.
            target_key: Target percentile key.
        """
        if not path:
            return
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        header = ["timestamp", "clip_percentile", "zero_shift", "layer", "layer_percentile", "adjusted_clip", "severity", "target_key"]
        file_exists = path_obj.exists()
        with open(path_obj, "a", newline="") as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(header)
            timestamp = time.time()
            target_percentile_value = self._percentile_name_to_value(target_key)
            for layer_idx, entry in sorted(schedule.items()):
                writer.writerow(
                    [
                        timestamp,
                        clip_pct,
                        zero_shift,
                        layer_idx,
                        target_percentile_value,
                        entry.get("clip_percentile"),
                        entry.get("severity"),
                        target_key,
                    ]
                )

    def find_top_k_layers(self, stats: Dict[int, Dict[str, float]], percentile_key: str, top_k: int = 8) -> list[int]:
        """Return top_k layer indices sorted by descending percentile value.

        Args:
            stats: Layer stats.
            percentile_key: Key like 'p9999'.
            top_k: Number of top layers.

        Returns:
            List of layer indices.
        """
        vals = []
        for layer, s in stats.items():
            val = s.get(percentile_key, 0.0)
            vals.append((layer, val))
        vals.sort(key=lambda x: x[1], reverse=True)
        return [l for l, _ in vals[:top_k]]

    @staticmethod
    @staticmethod
    def _percentile_name_to_value(name: str) -> float | None:
        """Convert percentile key to value, e.g., 'p9999' -> 0.9999."""
        if name.startswith("p") and name[1:].isdigit():
            return float(int(name[1:])) / 10000.0
        return None