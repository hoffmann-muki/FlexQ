"""OACS tuning harness

Profiles down-projection activations, identifies the top-K sensitive layers,
and runs a small grid sweep of clipping percentiles and zero-point shift scales.

Usage examples:
python algorithm/oacs_tuning.py --model ../models/llama-2-7b-hf --dataset wikitext2 --nsamples 16 --top_k 8

This script uses the adaptive flags that were added to `main.py` to enable the OACS logic inside the quantizer.
"""

import argparse
import csv
import time
import math
from pathlib import Path
import sys
import functools
import copy

import torch
import torch.nn as nn
from typing import Sequence

from algorithm.datautils import get_loaders
from algorithm.analysis.activation_stats import ActivationStatsHook
from algorithm.models.LMClass import LMClass
from algorithm.flexq_quantize.flexqllm import flexqllm
from algorithm.main import evaluate
from algorithm import utils
from algorithm.clipping_module import ClippingModule

sys.path.append(str(Path(__file__).parent / "duquant_integration"))
from algorithm.duquant_integration.quantize.duquant import duquant
import logging

logger = logging.getLogger(__name__)


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

    logger.info(f"Profiling up to {max_batches} batches on {len(hooks)} layers (device={device})...")
    batches_run = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            logger.info(f"  Running profiling batch {batch_idx + 1}/{max_batches}...")
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
            logger.debug(f"    Completed batch {batches_run}")

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


def run_sweep(
    args,
    layer_candidates,
    percentiles_to_try,
    zero_shifts_to_try,
    layer_profiles,
    channel_maxes=None,
    enable_clipping=False,
):
    clipper = ClippingModule() if enable_clipping else None

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
            if enable_clipping:
                layer_schedule = clipper.build_layer_clip_schedule(
                    layer_profiles,
                    clip_pct,
                    target_key,
                    args.layer_scheduler_bonus_scale,
                    args.layer_scheduler_bonus_cap,
                    zero_scale,
                )
                clipper.log_layer_schedule(
                    layer_schedule,
                    clip_pct,
                    zero_scale,
                    args.layer_schedule_path,
                    target_key,
                )
            else:
                layer_schedule = {}

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
            a.adaptive_clip_down_proj = enable_clipping
            a.adaptive_clip_percentile = clip_pct if enable_clipping else 1.0
            a.adaptive_zero_shift_scale = zero_scale if enable_clipping else 0.0
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
            a.p_quant_params = {"n_bits": 16, "metric": "fix0to1", "rotate": False}

            lm = LMClass(a)
            lm.seqlen = 2048
            lm.model.eval()
            for p in lm.model.parameters():
                p.requires_grad = False
            
            # ensured above; remove duplicate
            if args.enable_smoothing and channel_maxes is not None:
                apply_smoothing(lm, channel_maxes, alpha=args.smoothing_alpha, sensitive_layers=layer_candidates)

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
            logger.info(
                f"Tried clip_pct={clip_pct}, zero_scale={zero_scale} => PPL={ppl}  task_metrics={metrics_str} (took {elapsed:.1f}s)"
            )


def apply_smoothing(lm, channel_maxes, alpha=0.5, sensitive_layers: list | None = None):
    """
    Apply SmoothQuant-like smoothing to down_proj layers.
    Scales down activations (by scaling up up_proj weights) and scales up down_proj weights.
    """
    logger.info(f"Applying smoothing to down_proj layers (alpha={alpha})...")
    if hasattr(lm.model.model, "layers"):
        layer_list = lm.model.model.layers
    elif hasattr(lm.model.model, "decoder") and hasattr(lm.model.model.decoder, "layers"):
        layer_list = lm.model.model.decoder.layers
    else:
        return

    for i, lyr in enumerate(layer_list):
        # If sensitive_layers is provided, only apply smoothing to those layers
        if sensitive_layers is not None and i not in sensitive_layers:
            continue
        if i not in channel_maxes or channel_maxes[i] is None:
            continue
        
        # act_max is [intermediate_size]
        act_max = channel_maxes[i]
        # weight_max is [intermediate_size, hidden_size]
        weight_max = lyr.mlp.down_proj.weight.data.abs().max(dim=1)[0]
        
        # SmoothQuant scaling
        scale = (act_max ** alpha) / (weight_max ** (1 - alpha))
        scale = scale / scale.max()  # normalize
        
        # Apply to up_proj (scale up weights)
        lyr.mlp.up_proj.weight.data *= scale.unsqueeze(1)
        # Apply to down_proj (scale down weights)
        lyr.mlp.down_proj.weight.data /= scale.unsqueeze(0)


def get_act_scales(model, dataloader, num_samples=16):
    """Collect activation scales for DuQuant calibration."""
    act_scales = {}
    
    def stat_input_hook(name, x):
        if isinstance(x, tuple):
            x = x[0]
        x = x.detach().float()
        stat_tensor(name, x)

    def stat_tensor(name, x):
        if name not in act_scales:
            act_scales[name] = {"max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
        act_scales[name]["max"] = max(act_scales[name]["max"], x.abs().max().item())
        act_scales[name]["mean"] += x.abs().mean().item()
        act_scales[name]["std"] += x.abs().std().item()
        act_scales[name]["count"] += 1

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    logger.info("Collecting activation scales for DuQuant...")
    def _get_inputs_from_batch(batch):
        # Attempt several heuristics in order to obtain input_ids tensor or first tensor in batch
        try:
            if isinstance(batch, (list, tuple)):
                cand = batch[0]
            elif isinstance(batch, dict):
                if "input_ids" in batch:
                    cand = batch["input_ids"]
                else:
                    # fallback to first tensor-like value
                    for v in batch.values():
                        if torch.is_tensor(v):
                            cand = v
                            break
                    else:
                        cand = next(iter(batch.values()))
            elif hasattr(batch, "input_ids"):
                cand = batch.input_ids
            else:
                cand = batch
        except Exception:
            cand = batch
        return cand

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = _get_inputs_from_batch(batch)
            if torch.is_tensor(inputs):
                bs = inputs.shape[0]
            else:
                # if inputs isn't a tensor, try to move and let the model handle it
                try:
                    inputs = torch.tensor(inputs)
                    bs = inputs.shape[0]
                except Exception:
                    bs = 1
            if i * bs >= num_samples:
                break
            if torch.is_tensor(inputs):
                inputs = inputs.to(next(model.parameters()).device)
            else:
                # best-effort: try to coerce to tensor and move
                try:
                    inputs = torch.as_tensor(inputs).to(next(model.parameters()).device)
                except Exception:
                    # fallback: attempt to call model on this batch directly
                    inputs = inputs

            model(inputs)
    
    for h in hooks:
        h.remove()
        
    return act_scales


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
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate perplexity on Wikitext2 during the run")
    parser.add_argument("--enable_smoothing", action="store_true", help="Enable SmoothQuant-like activation smoothing")
    parser.add_argument("--smoothing_alpha", type=float, default=0.5, help="Alpha parameter for smoothing (0.5 balances weights and activations)")

    # DuQuant arguments
    parser.add_argument("--duquant", action="store_true", help="Enable DuQuant method")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_rotation_step", type=int, default=1024)
    parser.add_argument("--permutation_times", type=int, default=1)
    parser.add_argument("--swc", type=float, default=None)
    parser.add_argument("--lac", type=float, default=None)
    parser.add_argument("--lwc", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--let", action="store_true")
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--aug_loss", action="store_true")
    parser.add_argument("--smooth_epochs", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_alpha", type=float, default=0.5)
    parser.add_argument("--deactive_amp", action="store_true")
    parser.add_argument("--multigpu", action="store_true", help="Map model across multiple GPUs for evaluation")
    parser.add_argument("--resume", type=str, default=None, help="Path to duquant_parameters to resume from")
    parser.add_argument("--save_dir", default=None, type=str, help="Directory to save duquant params / artifacts")
    parser.add_argument("--quant_method", type=str, default="duquant", help="Quantization method to employ for DuQuant flow")
    parser.add_argument("--smooth", action="store_true", help="Enable Smooth Quant learning & params")

    parser.add_argument("--enable_clipping", action="store_true", help="Enable adaptive clipping for OACS. Disabled by default.")

    args = parser.parse_args()

    # ensure compatibility with LMClass which expects this arg
    if not hasattr(args, "attn_implementation"):
        args.attn_implementation = "eager"

    if args.device_map is None:
        args.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Defaulting --device_map to {args.device_map}")

    # HuggingFace requires `low_cpu_mem_usage=True` when passing a device_map.
    # If the user supplied a device_map but didn't enable the low-memory flag,
    # enable it automatically to avoid from_pretrained errors.
    if getattr(args, "device_map", None) is not None and not getattr(args, "low_cpu_mem_usage", False):
        logger.info("Note: enabling --low_cpu_mem_usage because --device_map was provided")
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
                logger.warning(f"Warning: unknown torch dtype '{td}', defaulting to torch.float16")
                args.torch_dtype = torch.float16

    # ensure cache directory exists (used by evaluate)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    args.layer_stats_path = args.layer_stats_path or None
    args.layer_schedule_path = args.layer_schedule_path or None

    # get a small calibration loader
    dataloader, testloader = get_loaders(args.dataset, seed=args.seed, model=args.model, seqlen=2048)

    if args.duquant:
        print("Running DuQuant...")
        lm = LMClass(args)
        lm.seqlen = 2048
        
        # Collect scales
        act_scales = get_act_scales(lm.model, dataloader, num_samples=args.nsamples)
        act_shifts = {} # We don't use shifts for now or assume 0
        
        # Prepare args for duquant
        # duquant expects args to have specific attributes
        args.net = args.model
        # DuQuant's UniformAffineQuantizer has a slightly different parameter
        # signature. Build a DuQuant-compatible params dict and avoid fields
        # unsupported by the DuQuant quantizer (e.g., disable_zero_point).
        args.weight_quant_params = {
            "n_bits": args.wbits,
            "per_channel_axes": [0],
            "symmetric": args.symmetric,
            "dynamic_method": args.w_dynamic_method,
            "group_size": args.w_group_size,
            "quant_method": "duquant",
            "block_size": args.block_size,
            "max_rotation_step": args.max_rotation_step,
            "permutation_times": args.permutation_times,
        }
        args.act_quant_params = {
            "n_bits": args.abits,
            "per_channel_axes": [],
            "symmetric": False,
            "dynamic_method": args.a_dynamic_method,
            "quant_method": "duquant",
            "block_size": args.block_size,
            "max_rotation_step": args.max_rotation_step,
            "permutation_times": args.permutation_times,
        }
        args.q_quant_params = copy.deepcopy(args.act_quant_params)
        args.k_quant_params = copy.deepcopy(args.act_quant_params)
        args.v_quant_params = copy.deepcopy(args.act_quant_params)
        # pv quantization (for attention values/attention weights) should not apply DuQuant rotations
        args.p_quant_params = copy.deepcopy(args.act_quant_params)
        args.p_quant_params["rotate"] = False
        args.p_quant_params["quant_method"] = None
        
        args.q_weight_quant_params = copy.deepcopy(args.weight_quant_params)
        args.k_weight_quant_params = copy.deepcopy(args.weight_quant_params)
        args.v_weight_quant_params = copy.deepcopy(args.weight_quant_params)
        args.o_weight_quant_params = copy.deepcopy(args.weight_quant_params)
        args.gate_weight_quant_params = copy.deepcopy(args.weight_quant_params)
        args.up_weight_quant_params = copy.deepcopy(args.weight_quant_params)
        args.down_weight_quant_params = copy.deepcopy(args.weight_quant_params)
        
        args.q_act_quant_params = copy.deepcopy(args.act_quant_params)
        args.k_act_quant_params = copy.deepcopy(args.act_quant_params)
        args.v_act_quant_params = copy.deepcopy(args.act_quant_params)
        args.o_act_quant_params = copy.deepcopy(args.act_quant_params)
        args.gate_act_quant_params = copy.deepcopy(args.act_quant_params)
        args.up_act_quant_params = copy.deepcopy(args.act_quant_params)
        args.down_act_quant_params = copy.deepcopy(args.act_quant_params)

        # Ensure args has fields expected by evaluate()
        if not hasattr(args, 'net') or args.net is None:
            args.net = args.model
        if not hasattr(args, 'model_family') or args.model_family is None:
            args.model_family = args.net.split("-")[0]
        if not hasattr(args, 'multigpu'):
            args.multigpu = False
        if not hasattr(args, 'eval_ppl'):
            args.eval_ppl = True

        # Run DuQuant
        duquant(lm, args, dataloader, act_scales, act_shifts, logger=utils.create_logger(Path(args.output_dir)))
        
        # Evaluate
        logger.debug("[DEBUG] Args keys before evaluate: %s", sorted(list(vars(args).keys())))
        results = evaluate(lm, args, utils.create_logger(Path(args.output_dir)))
        ppl = results.get("wikitext2", None)
        logger.info(f"DuQuant Results: PPL={ppl}")
        logger.info(results)
        return

    # we will use dataloader (train) for profiling
    logger.info("Profiling activations to find sensitive layers...")
    lm = LMClass(args)
    lm.seqlen = 2048
    profiles, channel_maxes = profile_down_proj_layers(lm, dataloader, percentiles=tuple(args.layer_stats_percentiles), max_batches=args.max_profile_batches)

    # choose top-k by p9999 if available else p999
    key = "p9999" if any("p9999" in s for s in profiles.values()) else "p999"
    if args.enable_clipping:
        clipper = ClippingModule()
        top_k_layers = clipper.find_top_k_layers(profiles, key, top_k=args.top_k)
    else:
        top_k_layers = find_top_k_layers(profiles, key, top_k=args.top_k)
    logger.info("Top-k sensitive layers: %s", top_k_layers)

    # Apply smoothing only to the identified sensitive layers when requested
    if args.enable_smoothing:
        apply_smoothing(lm, channel_maxes, alpha=args.smoothing_alpha, sensitive_layers=top_k_layers)

    log_layer_stats(profiles, args.layer_stats_percentiles, args.layer_stats_path)

    # If clipping is disabled, use single baseline values
    if not args.enable_clipping:
        percentiles_to_try = [1.0]
        zero_scales_to_try = [0.0]
    else:
        percentiles_to_try = args.percentiles
        zero_scales_to_try = args.zero_scales

    # run a small grid sweep using these top layers as candidates (we pass list for record)
    run_sweep(args, top_k_layers, percentiles_to_try, zero_scales_to_try, profiles, channel_maxes, args.enable_clipping)


if __name__ == "__main__":
    main()

