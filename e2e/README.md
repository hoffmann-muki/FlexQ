# FasterTransformer(FlexQ Version)

[FasterTransformer](https://github.com/NVIDIA/FasterTransformer) provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA. To evaluate end-to-end latency, this codebase is modified from [bytedance/ABQ-LLM/tree/main/fastertransformer](https://github.com/bytedance/ABQ-LLM/tree/main/fastertransformer).

Note that current codebase is for efficiency evaluation. We use random weights therefore no meaningful output.

## FasterTransformer E2E Evaluation

Please complete the FasterTransformer compilation (Make sure you install MPI):
```
```markdown
# FasterTransformer — End-to-end evaluation (FlexQ)

This directory contains a fork of the FasterTransformer evaluation harness adapted for FlexQ performance experiments. It is intended for latency and throughput measurements; the reference build may use synthetic or random weights for benchmarking.

Build

1. Ensure MPI and required build tools are installed on your system.
2. From the repository root run:

```bash
cd e2e
bash build.sh
```

Configuration

- Edit the appropriate model config file for your test: for LLaMA use `e2e/examples/cpp/llama/llama_config.ini`; for OPT use `e2e/examples/cpp/multi_gpu_gpt/gpt_config.ini`.
- Precision mapping used by the examples (config field `int8_mode`):
	- `0` = FP16
	- `1` = W8A16 (CUTLASS)
	- `2` = W8A8 (SmoothQuant)
	- `5` = W6Ax (FlexQ)
- For multi-GPU runs, set `tensor_para_size` to the number of GPUs.

Run example

From the build output directory (typically `build_release`):

```bash
# single-GPU LLaMA example
./bin/llama_example

# single-GPU OPT example
./bin/multi_gpu_gpt_example

# multi-GPU example (MPI)
mpirun -n 2 ./bin/llama_example
```

Notes

- This harness focuses on performance metrics. For functional accuracy or end-to-end evaluation with real weights and datasets, replace synthetic random weights with a real model checkpoint and provide appropriate input data.
- Check the example config files and the `examples` subdirectories for detailed options and model-specific parameters.
```
