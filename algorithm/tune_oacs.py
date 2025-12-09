"""OACS tuning harness

Profiles down-projection activations, identifies the top-K sensitive layers,
and runs a small grid sweep of clipping percentiles and zero-point shift scales.

Usage examples:
python algorithm/tune_oacs.py --model ../models/llama-2-7b-hf --dataset wikitext2 --nsamples 16 --top_k 8

This script uses the adaptive flags that were added to `main.py` to enable the OACS logic inside the quantizer.
"""

import argparse
import csv
import time
import math
from pathlib import Path

import torch
from typing import Sequence

from algorithm.datautils import get_loaders
from algorithm.analysis.activation_stats import ActivationStatsHook
from algorithm.models.LMClass import LMClass
from algorithm.flexq_quantize.flexqllm import flexqllm
from algorithm.main import evaluate
from algorithm import utils


def profile_down_proj_layers(lm, dataloader, percentiles=(0.999, 0.9999), max_batches=8, device=None):
    """Register hooks on `mlp.down_proj` of each decoder layer and run a few batches.
    Returns a dict mapping layer index -> summary dict of percentile stats.
    """
    hooks = []
    handles = []
    layers = []
    # assume Llama-like model
    if hasattr(lm.model.model, "layers"): 
        layer_list = lm.model.model.layers
    elif hasattr(lm.model.model, "decoder") and hasattr(lm.model.model.decoder, "layers"):
        layer_list = lm.model.model.decoder.layers
    else:
        raise RuntimeError("Unable to find model layers to attach hooks")

    for i, lyr in enumerate(layer_list):
        if hasattr(lyr, "mlp") and hasattr(lyr.mlp, "down_proj"):
            # Capture INPUT to down_proj for smoothing and correct OACS profiling
            hook = ActivationStatsHook(name=f"layer_{i}", percentiles=list(percentiles), capture_input=True)
            handle = lyr.mlp.down_proj.register_forward_hook(hook)
            hooks.append(hook)
            handles.append(handle)
            layers.append(i)

    # Run a few batches through the model
    lm.model.eval()
    old_device = lm._device if hasattr(lm, "_device") else None
    if device is None:
        device = lm.device
    lm.model.to(device)

    print(f"Profiling up to {max_batches} batches on {len(hooks)} layers (device={device})...")
    batches_run = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            print(f"  Running profiling batch {batch_idx + 1}/{max_batches}...")
            # dataloader yields (input_ids, attention_mask) or similar; try to find input ids
            if hasattr(batch, "input_ids"):
                inputs = batch.input_ids.to(device)
            elif isinstance(batch, (list, tuple)):
                # try to find a tensor or an object with `input_ids` inside the tuple/list
                found = None
                for elem in batch:
                    if hasattr(elem, "input_ids"):
                        found = elem.input_ids
                        break
                    if torch.is_tensor(elem):
                        found = elem
                        break
                    if isinstance(elem, dict) and "input_ids" in elem:
                        found = elem["input_ids"]
                        break
                if found is None:
                    raise RuntimeError("Unable to find input tensor in batch tuple/list")
                inputs = found.to(device)
            elif isinstance(batch, dict) and "input_ids" in batch:
                inputs = batch["input_ids"].to(device)
            else:
                # assume batch is tensor
                inputs = batch.to(device)

            # forward through model
            if "decoder" in lm.model.__class__.__name__.lower():
                _ = lm.model(inputs)
            else:
                try:
                    _ = lm.model(inputs)
                except Exception:
                    # fallback to calling model.model
                    _ = lm.model.model(inputs)
            batches_run += 1
            print(f"    Completed batch {batches_run}")

    # collect summaries
    summaries = {}
    channel_maxes = {}
    for hook, layer_idx in zip(hooks, layers):
        summaries[layer_idx] = hook.summary()
        channel_maxes[layer_idx] = hook.per_channel_max

    # remove hooks
    for h in handles:
        h.remove()

    # restore device
    if old_device is not None:
        lm._device = old_device

    return summaries, channel_maxes


def percentile_key(percentile: float) -> str:
    return f"p{int(percentile * 10000)}"

def percentile_name_to_value(name: str) -> float | None:
    if name.startswith("p") and name[1:].isdigit():
        return float(int(name[1:])) / 10000.0
    return None


def log_layer_stats(stats: dict[int, dict[str, float]], percentiles: Sequence[float], path: str | None) -> None:
    if not path:
        return
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "layer", "percentile", "stat", "value"]
    file_exists = path_obj.exists()
    with open(path_obj, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(header)
        timestamp = time.time()
        for layer_idx, layer_stats in sorted(stats.items()):
            logged_keys = set()
            for pct in percentiles:
                key = percentile_key(pct)
                if key in layer_stats:
                    writer.writerow([timestamp, layer_idx, percentile_name_to_value(key), key, layer_stats[key]])
                    logged_keys.add(key)
            for key, value in sorted(layer_stats.items()):
                if key in logged_keys:
                    continue
                writer.writerow([timestamp, layer_idx, percentile_name_to_value(key), key, value])


def build_layer_clip_schedule(
    stats: dict[int, dict[str, float]],
    base_clip_pct: float,
    target_percentile_key: str,
    bonus_scale: float,
    bonus_cap: float,
    zero_shift: float,
) -> dict[int, dict[str, float]]:
    schedule: dict[int, dict[str, float]] = {}
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
    schedule: dict[int, dict[str, float]],
    clip_pct: float,
    zero_shift: float,
    path: str | None,
    target_key: str,
) -> None:
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
        target_percentile_value = percentile_name_to_value(target_key)
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


def find_top_k_layers(stats: dict, percentile_key: str, top_k: int = 8):
    """Return top_k layer indices sorted by descending percentile value."""
    vals = []
    for layer, s in stats.items():
        val = s.get(percentile_key, 0.0)
        vals.append((layer, val))
    vals.sort(key=lambda x: x[1], reverse=True)
    return [l for l, _ in vals[:top_k]]


def run_sweep(
    args,
    layer_candidates,
    percentiles_to_try,
    zero_shifts_to_try,
    layer_profiles,
    channel_maxes=None,
):
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "timestamp",
        "clip_percentile",
        "zero_shift_scale",
        "top_k_layers",
        "ppl",
        "task_metrics",
    ]
    if not out_path.exists():
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)

    target_key = percentile_key(args.layer_scheduler_target_percentile)

    for clip_pct in percentiles_to_try:
        for zero_scale in zero_shifts_to_try:
            layer_schedule = build_layer_clip_schedule(
                layer_profiles,
                clip_pct,
                target_key,
                args.layer_scheduler_bonus_scale,
                args.layer_scheduler_bonus_cap,
                zero_scale,
            )
            log_layer_schedule(
                layer_schedule,
                clip_pct,
                zero_scale,
                args.layer_schedule_path,
                target_key,
            )

            t0 = time.time()
            class A:
                pass

            a = A()
            a.model = args.model
            a.cache_dir = args.cache_dir
            a.output_dir = args.output_dir
            a.save_dir = None
            a.calib_dataset = args.dataset
            a.nsamples = args.nsamples
            a.batch_size = args.batch_size
            a.seed = args.seed
            a.tasks = args.tasks
            a.eval_ppl = True
            a.num_fewshot = args.num_fewshot
            a.limit = args.limit
            a.multigpu = False
            a.deactive_amp = True
            a.attn_implementation = "eager"
            a.net = args.net if args.net is not None else Path(args.model).name
            a.model_family = a.net.split("-")[0]
            a.wbits = args.wbits
            a.w_group_size = args.w_group_size
            a.abits = args.abits
            a.a_group_size = args.a_group_size
            a.symmetric = args.symmetric
            a.disable_zero_point = args.disable_zero_point
            a.a_dynamic_method = args.a_dynamic_method
            a.w_dynamic_method = args.w_dynamic_method
            a.flex_linear_quant = args.flex_linear_quant
            a.device_map = args.device_map
            a.low_cpu_mem_usage = args.low_cpu_mem_usage
            a.torch_dtype = args.torch_dtype
            a.adaptive_clip_down_proj = True
            a.adaptive_clip_percentile = clip_pct
            a.adaptive_zero_shift_scale = zero_scale
            a.layer_clip_schedule = layer_schedule

            a.weight_quant_params = {
                "n_bits": a.wbits,
                "per_channel_axes": [0],
                "symmetric": a.symmetric,
                "dynamic_method": a.w_dynamic_method,
                "group_size": a.w_group_size,
                "disable_zero_point": a.disable_zero_point,
            }

            a.act_quant_params = {
                "n_bits": a.abits if a.flex_linear_quant is False else 6,
                "per_channel_axes": [],
                "symmetric": False,
                "dynamic_method": a.a_dynamic_method,
            }
            a.act_down_proj_quant_params = {
                "n_bits": a.abits if a.flex_linear_quant is False else 8,
                "per_channel_axes": [],
                "symmetric": False,
                "dynamic_method": a.a_dynamic_method,
            }

            if a.a_group_size:
                a.act_quant_params = {
                    "n_bits": a.abits if a.flex_linear_quant is False else 6,
                    "per_channel_axes": [],
                    "symmetric": a.symmetric,
                    "dynamic_method": a.a_dynamic_method,
                    "group_size": a.a_group_size,
                    "disable_zero_point": a.disable_zero_point,
                }
                a.act_down_proj_quant_params = {
                    "n_bits": a.abits if a.flex_linear_quant is False else 8,
                    "per_channel_axes": [],
                    "symmetric": a.symmetric,
                    "dynamic_method": a.a_dynamic_method,
                    "group_size": a.a_group_size,
                    "disable_zero_point": a.disable_zero_point,
                }

            if a.adaptive_clip_down_proj:
                a.act_down_proj_quant_params.update({
                    "clip_percentile": a.adaptive_clip_percentile,
                    "zero_shift_scale": a.adaptive_zero_shift_scale,
                })

            a.q_quant_params = {"n_bits": 16, "per_channel_axes": [], "symmetric": a.symmetric, "dynamic_method": a.a_dynamic_method}
            a.k_quant_params = {"n_bits": 16, "per_channel_axes": [], "symmetric": a.symmetric, "dynamic_method": a.a_dynamic_method}
            a.v_quant_params = {"n_bits": 16, "per_channel_axes": [], "symmetric": a.symmetric, "dynamic_method": a.a_dynamic_method}
            a.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}

            lm = LMClass(a)
            lm.seqlen = 2048
            lm.model.eval()
            for p in lm.model.parameters():
                p.requires_grad = False
            
            # Apply smoothing if enabled (using the channel_maxes collected during profiling)
            if args.enable_smoothing and channel_maxes is not None:
                apply_smoothing(lm, channel_maxes, alpha=args.smoothing_alpha)

            flexqllm(lm, a, utils.create_logger(Path(a.output_dir)))

            results = evaluate(lm, a, utils.create_logger(Path(a.output_dir)))
            ppl = results.get("wikitext2", None)
            task_metrics = []
            for task, metrics in results.get("results", {}).items():
                for metric, value in metrics.items():
                    if metric.endswith("_stderr"):
                        continue
                    if isinstance(value, float):
                        task_metrics.append(f"{task}.{metric}={value:.6g}")
                    else:
                        task_metrics.append(f"{task}.{metric}={value}")
            metrics_str = ";".join(task_metrics)

            with open(out_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([time.time(), clip_pct, zero_scale, ";".join(map(str, layer_candidates)), ppl, metrics_str])

            elapsed = time.time() - t0
            print(
                f"Tried clip_pct={clip_pct}, zero_scale={zero_scale} => PPL={ppl}  task_metrics={metrics_str} (took {elapsed:.1f}s)"
            )


def apply_smoothing(lm, channel_maxes, alpha=0.5):
    """
    Apply SmoothQuant-like smoothing to down_proj layers.
    Scales down activations (by scaling up up_proj weights) and scales up down_proj weights.
    """
    print(f"Applying smoothing to down_proj layers (alpha={alpha})...")
    if hasattr(lm.model.model, "layers"): 
        layer_list = lm.model.model.layers
    elif hasattr(lm.model.model, "decoder") and hasattr(lm.model.model.decoder, "layers"):
        layer_list = lm.model.model.decoder.layers
    else:
        return

    for i, lyr in enumerate(layer_list):
        if i not in channel_maxes or channel_maxes[i] is None:
            continue
        
        # act_max is [intermediate_size]
        act_max = channel_maxes[i].to(lyr.mlp.down_proj.weight.device)
        
        # w_max is [intermediate_size] (max over hidden_size dim)
        # down_proj.weight is [hidden, interm]
        w_max = lyr.mlp.down_proj.weight.abs().max(dim=0).values
        
        # Avoid division by zero
        act_max = act_max.clamp(min=1e-5)
        w_max = w_max.clamp(min=1e-5)
        
        # Calculate scale: s = act_max^alpha / w_max^(1-alpha)
        # With alpha=0.05, we saw scales ~12.0, which is still too high for W6.
        # We need to dampen the scale significantly.
        # Let's try a much gentler scaling: s = (act_max / w_max)^alpha
        # This is mathematically equivalent to the previous formula if we just change alpha,
        # but let's be explicit about the goal: we want s to be close to 1.0.
        
        # Current observation: act_max >> w_max, so ratio is large (e.g. 100).
        # 100^0.05 = 1.25. 
        # But we saw mean=12.0? That implies ratio is HUGE (e.g. 10^20?) or my math is off.
        # Wait, previous formula was: s = act^alpha / w^(1-alpha)
        # If act=100, w=0.1, alpha=0.05:
        # s = 100^0.05 / 0.1^0.95 = 1.25 / 0.11 = 11.3
        # This explains why we got ~11-12.
        
        # NEW FORMULA: s = (act_max / w_max).pow(alpha)
        # If act=100, w=0.1, alpha=0.05 => (1000)^0.05 = 1.41
        # This is much safer.
        
        ratio = (act_max / w_max).clamp(min=1e-5)
        scale = ratio.pow(alpha).clamp(min=1e-5)
        
        # Safety: Clamp scale to be within [0.5, 2.0] to prevent any extreme distortion
        scale = scale.clamp(min=0.5, max=2.0)
        
        print(f"  Layer {i}: scale min={scale.min():.4f}, max={scale.max():.4f}, mean={scale.mean():.4f}")

        # Apply scale:
        # input' = input / scale
        # weight' = weight * scale
        
        # Scale down_proj weights (weight * scale)
        # weight is [hidden, interm], scale is [interm]
        lyr.mlp.down_proj.weight.data.mul_(scale.view(1, -1))
        
        # Scale up_proj weights (weight / scale) to produce input / scale
        # up_proj weight is [interm, hidden], scale is [interm]
        lyr.mlp.up_proj.weight.data.div_(scale.view(-1, 1))
        
    print("Smoothing applied successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="../log/")
    parser.add_argument("--output_csv", type=str, default="algorithm/oacs_tuning_results.csv")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--percentiles", nargs='+', type=float, default=[0.995, 0.9975, 0.999, 0.9995, 0.9999])
    parser.add_argument("--zero_scales", nargs='+', type=float, default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--max_profile_batches", type=int, default=4)
    parser.add_argument(
        "--layer_stats_percentiles",
        nargs='+',
        type=float,
        default=[0.9, 0.95, 0.99, 0.999, 0.9999],
        help="percentiles gathered during layer profiling",
    )
    parser.add_argument(
        "--layer_stats_path",
        type=str,
        default="algorithm/oacs_layer_stats.csv",
        help="path to append per-layer stats; leave empty to disable",
    )
    parser.add_argument(
        "--layer_schedule_path",
        type=str,
        default="algorithm/oacs_layer_schedule.csv",
        help="path to append per-sweep layer schedule entries; leave empty to disable",
    )
    parser.add_argument(
        "--layer_scheduler_target_percentile",
        type=float,
        default=0.9,
        help="base percentile used to compare tail severity when building per-layer schedules",
    )
    parser.add_argument(
        "--layer_scheduler_bonus_scale",
        type=float,
        default=0.05,
        help="scale factor for clipping bonus based on tail severity",
    )
    parser.add_argument(
        "--layer_scheduler_bonus_cap",
        type=float,
        default=0.1,
        help="maximum additional percentile fraction allocated to bad layers",
    )
    parser.add_argument("--wbits", type=int, default=6)
    parser.add_argument("--abits", type=int, default=6)
    parser.add_argument("--w_group_size", type=int, default=None)
    parser.add_argument("--a_group_size", type=int, default=None)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--disable_zero_point", action="store_true")
    parser.add_argument("--flex_linear_quant", action="store_true")
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None,
        help="HuggingFace device_map to load the model on (defaults to GPU when available)")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="keep huggingface low memory flag when loading the model")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token", "per_group"]) 
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel", "per_group"]) 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="piqa")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--enable_smoothing", action="store_true", help="Enable SmoothQuant-like activation smoothing")
    parser.add_argument("--smoothing_alpha", type=float, default=0.5, help="Alpha parameter for smoothing (0.5 balances weights and activations)")

    args = parser.parse_args()

    # ensure compatibility with LMClass which expects this arg
    if not hasattr(args, "attn_implementation"):
        args.attn_implementation = "eager"

    if args.device_map is None:
        args.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Defaulting --device_map to {args.device_map}")

    # HuggingFace requires `low_cpu_mem_usage=True` when passing a device_map.
    # If the user supplied a device_map but didn't enable the low-memory flag,
    # enable it automatically to avoid from_pretrained errors.
    if getattr(args, "device_map", None) is not None and not getattr(args, "low_cpu_mem_usage", False):
        print("Note: enabling --low_cpu_mem_usage because --device_map was provided")
        args.low_cpu_mem_usage = True

    # Normalize torch_dtype: allow user to pass strings like 'float16' or 'auto'
    if isinstance(getattr(args, "torch_dtype", None), str):
        td = args.torch_dtype
        if td == "auto":
            args.torch_dtype = "auto"
        else:
            # map common names to torch dtype objects
            try:
                args.torch_dtype = getattr(torch, td)
            except Exception:
                print(f"Warning: unknown torch dtype '{td}', defaulting to torch.float16")
                args.torch_dtype = torch.float16

    # ensure cache directory exists (used by evaluate)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    args.layer_stats_path = args.layer_stats_path or None
    args.layer_schedule_path = args.layer_schedule_path or None

    # get a small calibration loader
    dataloader, testloader = get_loaders(args.dataset, seed=args.seed, model=args.model, seqlen=2048)

    # we will use dataloader (train) for profiling
    print("Profiling activations to find sensitive layers...")
    lm = LMClass(args)
    lm.seqlen = 2048
    profiles, channel_maxes = profile_down_proj_layers(lm, dataloader, percentiles=tuple(args.layer_stats_percentiles), max_batches=args.max_profile_batches)

    if args.enable_smoothing:
        apply_smoothing(lm, channel_maxes, alpha=args.smoothing_alpha)

    # choose top-k by p9999 if available else p999
    key = "p9999" if any("p9999" in s for s in profiles.values()) else "p999"
    top_k_layers = find_top_k_layers(profiles, key, top_k=args.top_k)
    print("Top-k sensitive layers:", top_k_layers)

    log_layer_stats(profiles, args.layer_stats_percentiles, args.layer_stats_path)

    # run a small grid sweep using these top layers as candidates (we pass list for record)
    run_sweep(args, top_k_layers, args.percentiles, args.zero_scales, profiles, channel_maxes)


if __name__ == "__main__":
    main()
