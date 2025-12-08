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
from pathlib import Path

import torch

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
            hook = ActivationStatsHook(name=f"layer_{i}", percentiles=list(percentiles))
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

    batches_run = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
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

    # collect summaries
    summaries = {}
    for hook, layer_idx in zip(hooks, layers):
        summaries[layer_idx] = hook.summary()

    # remove hooks
    for h in handles:
        h.remove()

    # restore device
    if old_device is not None:
        lm._device = old_device

    return summaries


def find_top_k_layers(stats: dict, percentile_key: str, top_k: int = 8):
    """Return top_k layer indices sorted by descending percentile value."""
    vals = []
    for layer, s in stats.items():
        val = s.get(percentile_key, 0.0)
        vals.append((layer, val))
    vals.sort(key=lambda x: x[1], reverse=True)
    return [l for l, _ in vals[:top_k]]


def run_sweep(args, layer_candidates, percentiles_to_try, zero_shifts_to_try):
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header
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

    for clip_pct in percentiles_to_try:
        for zero_scale in zero_shifts_to_try:
            t0 = time.time()
            # create fresh args-like namespace to pass into LMClass and quantizer
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
            # pass through the normalized torch dtype (either a torch.dtype or 'auto')
            a.torch_dtype = args.torch_dtype
            a.adaptive_clip_down_proj = True
            a.adaptive_clip_percentile = clip_pct
            a.adaptive_zero_shift_scale = zero_scale

            # prepare quantization parameter dicts (same logic as main.py)
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

            # instantiate model, quantize, evaluate
            lm = LMClass(a)
            lm.seqlen = 2048
            lm.model.eval()
            for p in lm.model.parameters():
                p.requires_grad = False

            # run quantization (in-place)
            flexqllm(lm, a, utils.create_logger(Path(a.output_dir)))

            # run evaluation (small scale). We call evaluate from main which returns a dict
            results = evaluate(lm, a, utils.create_logger(Path(a.output_dir)))
            ppl = results.get("wikitext2", None)
            task_metrics = []
            for task, metrics in results.get("results", {}).items():
                for metric, value in metrics.items():
                    if metric.endswith("_stderr"):
                        continue
                    # format numbers consistently
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
    parser.add_argument("--max_profile_batches", type=int, default=8)
    parser.add_argument("--wbits", type=int, default=6)
    parser.add_argument("--abits", type=int, default=6)
    parser.add_argument("--w_group_size", type=int, default=None)
    parser.add_argument("--a_group_size", type=int, default=None)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--disable_zero_point", action="store_true")
    parser.add_argument("--flex_linear_quant", action="store_true")
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--device_map", type=str, default="cpu")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="keep huggingface low memory flag when loading the model")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token", "per_group"]) 
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel", "per_group"]) 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="piqa")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()

    # ensure compatibility with LMClass which expects this arg
    if not hasattr(args, "attn_implementation"):
        args.attn_implementation = "eager"

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

    # get a small calibration loader
    dataloader, testloader = get_loaders(args.dataset, seed=args.seed, model=args.model, seqlen=2048)

    # we will use dataloader (train) for profiling
    print("Profiling activations to find sensitive layers...")
    lm = LMClass(args)
    lm.seqlen = 2048
    profiles = profile_down_proj_layers(lm, dataloader, percentiles=(0.999, 0.9999), max_batches=args.max_profile_batches)

    # choose top-k by p9999 if available else p999
    key = "p9999" if any("p9999" in s for s in profiles.values()) else "p999"
    top_k_layers = find_top_k_layers(profiles, key, top_k=args.top_k)
    print("Top-k sensitive layers:", top_k_layers)

    # run a small grid sweep using these top layers as candidates (we pass list for record)
    run_sweep(args, top_k_layers, args.percentiles, args.zero_scales)


if __name__ == "__main__":
    main()
