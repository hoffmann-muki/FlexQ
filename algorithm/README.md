# Algorithm: Uniform W6A6 Activation Pathway

This document describes the algorithmic approach and tooling to run a model in W6A6 (6-bit weights and activations) by reshaping activation distributions in identified sensitive layers.

## Motivation

Certain layers (notably MLP down-projections) exhibit rare, high-magnitude activation spikes that increase the dynamic range required by the activation quantizer. DuQuant is the primary method used in this codebase to enable a uniform W6A6 pathway by identifying and adaptively handling sensitive layers. Complementary, optional techniques — such as activation smoothing and percentile clipping with a small zero-point shift — can be applied alongside DuQuant (or other flows) when beneficial to improve stability.

## Key code locations

- `algorithm/models/int_llama_layer.py`: quantized layer wrappers (`QuantLlamaMLP.down_proj`) and wiring for activation parameters.
- `algorithm/flexq_quantize/quantizer.py`: `UniformAffineQuantizer` implementing per-token/group dynamic calibration and affine quantization.
- `algorithm/analysis/activation_stats.py`: profiling helpers to measure percentiles and maxima.

## Instrumentation and profiling

1. Install runtime dependencies and activate the algorithm environment:

```bash
conda activate duplexquant
cd algorithm
pip install -r requirements.txt
```

2. Run a synthetic check (no model required):

```bash
# runs a synthetic tensor sampler and prints percentiles/max
python algorithm/analysis/activation_stats.py --shape 4 128 4096 --percentiles 0.99 0.999 0.9999
```

3. Attach the profiler to a model module (example):

```py
from algorithm.analysis.activation_stats import ActivationStatsHook

# After constructing the LM and before calibration
hook = ActivationStatsHook(name='down_proj', percentiles=[0.99, 0.999])
handle = lm.model.layers[<layer_index>].mlp.down_proj.register_forward_hook(hook)
# run calibration batches
print(hook.summary())
handle.remove()
```

### Optional: percentile clipping and zero-point shift

Percentile clipping and a small zero-point shift are optional, per-layer operations that may be useful when DuQuant alone does not sufficiently reduce outlier influence. Recommended usage:

1. Implement `adaptive_clip_and_shift(x, percentile, zero_shift_scale)` to compute a percentile threshold, clip values (symmetric or asymmetric), and apply a small zero-point shift.
2. Integrate the wrapper into the activation path used by `QuantLinear` so clipping is applied consistently during calibration and inference.
3. Expose per-layer thresholds and zero-point parameters via `module.register_buffer` and persist them for inference.
4. Validate the change by measuring quantization error and downstream metrics; enable only for layers where it improves stability or accuracy.

## Calibration workflow (recommended)

1. Run a short calibration pass with realistic inputs (e.g., `wikitext-2` or a c4 sample) to collect statistics for candidate layers.
2. If using clipping, compute per-layer thresholds (e.g., 99.9th percentile) and small zero-point shifts; otherwise collect the per-layer stats required by DuQuant.
3. Store parameters via module buffers and update the quantizer to use them at inference.

## Evaluation and cautions

- Compare FP16 perplexity against W6A6-with-adaptive-clipping; expect a small accuracy degradation but aim for parity with previous W6A8-enabled runs.
- Start conservative (high percentile, small shifts); aggressive clipping can bias activations and harm accuracy.
- Validate changes thoroughly on held-out sets.
