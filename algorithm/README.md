# Algorithm: Uniform W6A6 Activation Pathway

This document describes the algorithm-level design and tooling required to remove the W6A8 fallback and run the whole model in W6A6 (6-bit weights, 6-bit activations) by reshaping activation distributions in so-called "sensitive" layers.

## Motivation

Certain layers (notably MLP down-projections) exhibit rare, high-magnitude activation spikes. Those spikes enlarge the dynamic range the activation quantizer must cover, forcing the codebase to fall back to W6A8 in those layers to avoid catastrophic quantization error. Instead of increasing activation precision, we reshape the activation distribution with two complementary operations:

- Percentile-based clipping: choose an upper threshold using a high percentile (e.g. 99.5–99.99%) of the absolute activation values rather than the absolute max.
- Calibrated zero-point shift: apply a small shift so the clipped window better centers the inlier mass, reducing asymmetric error.

Together these steps compress the outlier influence while keeping the main distribution intact, letting the existing INT6 quantizer handle activations uniformly.

## Where to look in the codebase

- Quantized layer wrapper: `algorithm/models/int_llama_layer.py` — `QuantLlamaMLP.down_proj` constructs the down-projection `QuantLinear` and receives `args.act_down_proj_quant_params` when `--flex_linear_quant` is enabled.
- Quantizer implementation: `algorithm/flexq_quantize/quantizer.py` — `UniformAffineQuantizer` implements per-token/group dynamic calibration and the affine quantization pipeline.
- Activation profiling utility: `algorithm/analysis/activation_stats.py` — helper hooks and summarizers used to measure percentile and maximum values.

If your goal is to remove `W6A8` usage, the changepoint is the wiring above: either remove the `flex_linear_quant` switch or replace its high-precision path with an adaptive clipping + zero-point wrapper that feeds into the INT6 quantizer.

## Instrumentation and profiling

1. Install runtime deps (use the repository's algorithm environment):

```bash
conda activate flexq
cd algorithm
pip install -r requirements.txt
```

2. Basic synthetic check (no model required):

```bash
# runs a synthetic tensor sampler and prints percentile/max
python algorithm/analysis/activation_stats.py --shape 4 128 4096 --percentiles 0.99 0.999 0.9999
```

3. Attach the profiler to a real model's module in Python (example snippet):

```py
from algorithm.analysis.activation_stats import ActivationStatsHook
from algorithm.models.int_llama_layer import QuantLlamaMLP

# Example: after you construct/wrap your LM and before running calibration
hook = ActivationStatsHook(name='down_proj', percentiles=[0.99, 0.999])
# locate the actual submodule and register the forward hook
handle = lm.model.layers[<layer_index>].mlp.down_proj.register_forward_hook(hook)

# run calibration batches through the model (use your usual calibration loader)
# e.g. flexq calibration loop or a few dataset samples

# inspect collected stats
print(hook.summary())
# when done, remove the hook
handle.remove()
```

Notes:
- Replace `lm.model.layers[<layer_index>]` with the actual path to the layer object in your model.
- The `ActivationStatsHook` records per-forward-call summaries; use `hook.summary()` to get averaged values across calibration input.

## Implementing percentile clipping + zero-point shift

Suggested implementation approach:

1. Create a small wrapper function (e.g. `adaptive_clip_and_shift(x, percentile, zero_shift_scale)`) that:
   - computes a percentile threshold from a running statistic or a calibration pass;
   - clips values to `[-threshold, threshold]` (or asymmetric bounds if needed);
   - computes a small zero-point (shift) based on an observed inlier mean and a tunable scale factor;
   - returns `x_clipped_shifted` ready for the existing INT6 quantizer.

2. Integrate the wrapper into the activation path used by `QuantLinear` for activations (preferably inside the quantizer wrapper so it is applied atomically with scale/zero computation).

3. Expose per-layer percentile and zero-point parameters in the calibration stage. Persist these parameters (e.g. in model config or as registered buffers) so they can be reused at inference.

4. Validate by measuring quantization error and downstream metrics: run the `activation_stats` profiler before and after applying clipping to confirm the 99.9th percentile and maxima moved inside the representable INT6 range.

## Calibration workflow (recommended)

1. Run a short calibration pass with realistic inputs (e.g. `datasets/wikitext-2-raw-v1` or a sample of `c4`) to collect percentiles for each candidate layer.
2. Compute per-layer thresholds (e.g. 99.9th percentile) and a per-layer zero-point shift (small, typically << scale).
3. Store thresholds and zero-points (e.g., as `module.register_buffer('clip_thr', thr)` and `module.register_buffer('zero_shift', zp)`).
4. Update the quantizer to use `clip_thr` and `zero_shift` at inference; keep the quantization bit-width fixed at 6 bits.

## Evaluation and regression checks

- Compare FP16 perplexity/metrics to the W6A6-with-adaptive-clipping run. Expect a small drop relative to full FP16, and aim for parity with previous W6A8-enabled runs.
- Re-run `algorithm/analysis/activation_stats.py` or registered hooks to confirm that the 99.9th percentile and max are within the INT6 representable range after clipping.

### Adaptive clipping knobs

The default build keeps the existing uniform INT6 path intact. To try the new outlier-aware path, enable the `--adaptive_clip_down_proj` flag when running `main.py` and specify the percentile/shift parameters you want to register with the quantizer. Those arguments get shipped into `act_down_proj_quant_params` and reach the `UniformAffineQuantizer` used by the down-projection `QuantLinear` wrapper.

Example:

```bash
python main.py \
   --model ../models/llama-2-7b-hf \
   --wbits 6 --abits 6 \
   --flex_linear_quant \
   --adaptive_clip_down_proj \
   --adaptive_clip_percentile 0.9995 \
   --adaptive_zero_shift_scale 0.05 \
   # other flags...
```

This lets you run both the legacy W6A8 fallback (`--flex_linear_quant`) and the new percentile-based clipping beside it so you can compare metrics without touching the uniform path.

## Implementation notes and cautions

- Start conservative: prefer a higher percentile (e.g. 99.9) and small shifts; aggressive clipping can bias activations and harm accuracy.
- Keep per-layer configuration where needed; group-level thresholds are an option to reduce calibration storage.
- This change is algorithmic — validate thoroughly on your held-out evaluation sets.
