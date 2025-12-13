<h1 align="center">DuplexQuant: Efficient Post-training INT6 Quantization for LLM Serving</h1>
DuplexQuant is a post-training INT6 quantization framework built on FlexQ that enables uniform W6A6 deployment for large language models. The project combines: (1) fine-grained weight and activation group quantization; (2) an algorithmic DuQuant flow that permits uniform 6-bit weights and activations across layers; (3) optional activation-smoothing (a light redistribution of scale between activations and weights) to improve robustness; and (4) system-level optimizations for efficient execution on common GPU architectures.

## Install
1. Clone this repository
```
git clone https://github.com/hoffmann-muki/DuplexQuant.git
cd DuplexQuant
```

2. Install runtime dependencies
```
conda create -n duplexquant python=3.10 -y
conda activate duplexquant
cd ./DuplexQuant/algorithm
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Recommended: reproducible FP16 evaluation workflow
-------------------------------------------------
For FP16 accuracy evaluation with Llama-2 models, follow these steps:

1. Create and activate the Conda environment and install requirements (see above).

2. Obtain model access from Hugging Face (for gated Meta Llama models) and optionally download a local snapshot using `huggingface_hub.snapshot_download` after logging in with `huggingface-cli login`.

3. Run a sample FP16 evaluation (model path can be local or Hugging Face ID):

```bash
cd ./DuplexQuant/algorithm
python main.py --model /path/to/local/models/llama-2-7b-hf --net Llama-2-7b --eval_ppl --deactive_amp
# or download on demand:
python main.py --model meta-llama/Llama-2-7b-hf --net Llama-2-7b --eval_ppl --deactive_amp
```

Keep `--wbits` and `--abits` at 16 for FP16 evaluation; quantization runs are enabled when these flags are set to lower values.

Portable environment options
----------------------------
You can keep your conda environment inside the project (a "prefix" environment) to avoid recreating a named environment every time you move hosts. Two common patterns:

- Prefix env (quick, local): create the environment inside the repository and install packages there.

```bash
# from the repository root
conda create --prefix ./duplexquant/.conda-env python=3.10 -y
conda activate ./duplexquant/.conda-env
pip install --upgrade pip setuptools wheel
pip install -r algorithm/requirements.txt
# install torch wheel appropriate for the host's CUDA (example):
pip install --index-url https://download.pytorch.org/whl/cu121/ torch==2.2.0
```

- Conda-pack (portable archive for identical hosts): use `conda-pack` to create a relocatable tarball of the prefix environment. Copy the tarball to an identical Linux host and run the bundled `conda-unpack` helper inside the unpacked folder to fix absolute prefixes.

```bash
conda install -c conda-forge conda-pack -y
conda-pack -p ./duplexquant/.conda-env -o ./duplexquant/duplexquant-conda-env.tar.gz
# on the destination host, extract and run:
tar -xzf duplexquant-conda-env.tar.gz -C <target_dir>
<target_dir>/bin/conda-unpack
```

Important: CUDA / PyTorch binary compatibility and system toolkits are still host-specific. The conda-packed environment is portable only between compatible Linux distributions/architectures with compatible drivers; PyTorch CUDA wheels may need to be reinstalled for a different driver/CUDA runtime.

## Usage
### Accuracy Evaluation
You can execute the following scripts to complete the **FP16** Accuracy Evaluation.
```
python main.py --model /Path/To/Model \
--eval_ppl --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```
You can execute the following scripts to complete the **FlexQ W6Ax** Accuracy Evaluation.
```
python main.py --model /Path/To/Model \
--wbits 6 --abits 6 --w_group_size 128 --a_group_size 128 \
--flex_linear_quant --symmetric \
--eval_ppl --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```
The following describes critical configuration parameters:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--abits`: activation quantization bits.
- `--group_size`: group size for weight/activation quantization. If unset, defaults to per-channel quantization.
- `--symmetric`: use symmetric quantization. If unset, defaults to asymmetric quantization.
- `--flex_linear_quant`: Enables the flexq uniform W6A6 pathway. When set, the code runs the calibration and transforms required to apply a uniform 6-bit activation and weight quantization pipeline (with optional activation-smoothing), removing the need for selective high-precision fallbacks.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--tasks`: evaluating zero-shot tasks.

DuQuant and Activation Smoothing
------------------------------------------
DuQuant enhances the uniform W6A6 pathway in this repository. It is an algorithmic calibration and transform pipeline that prepares activations and weights for uniform 6-bit representation. Activation-smoothing is a complementary technique that further improves robustness. Both features are opt-in. The design objective of DuQuant is to remove the need for selective high-precision activation fallbacks by reducing per-layer quantization error through algorithmic calibration and light per-layer transforms. Activation-smoothing is a controlled redistribution of scale between activations and weights (akin to SmoothQuant) that reduces range mismatches and improves stability for aggressive W6A6 quantization.

Operational guidance
- These features are opt-in. Start with conservative calibration settings (moderate calibration sample counts and limited per-layer transforms) and compare against FP16 baselines before enabling more aggressive transforms.
- Use the provided profiling tools to validate calibration results and confirm that per-layer statistics indicate safe uniform quantization.
- If instability is observed (e.g., large per-layer MSE or NaNs), reduce transform strength or increase calibration samples; the tooling supports safe fallback to less aggressive configurations.

Key command-line knobs (examples)
- Enable the DuQuant flow with the corresponding CLI flag and control learning-based tuning via `--let` and `--lwc` (plus `--epochs` and learning rates). Use `--nsamples` to set the number of calibration samples.
- Enable activation smoothing with a flag such as `--enable_smoothing` and tune strength with `--smoothing_alpha`.

These additions are intended to be used by practitioners familiar with post-training quantization workflows; consult the code comments and the `algorithm/` directory for the available CLI flags and profiling utilities.

### Kernel Benchmark
Please complete the compilation of the FlexQ kernel first:
```
cd ./DuplexQuant/engine
bash build.sh
```
To obtain benchmark results for the cuBLAS(W8A8) kernel, please execute:
```
bash test_cublas_kernel.sh
```
To obtain benchmark results for the FlexQ kernel, please execute:
```
bash test_flexq_kernel.sh
```

### FasterTransformer E2E Performance
Please complete the FasterTransformer compilation (Make sure you install MPI):
```
cd ./DuplexQuant/e2e
bash build.sh
``` 

Modify the evaluation configuration:
```
# For LLaMA model, modify: e2e/examples/cpp/llama/llama_config.ini
# For OPT model, modify: e2e/examples/cpp/multi_gpu_gpt/gpt_config.ini

The following are the precision parameter settings for different baselines:
FP16:               int8_mode=0
W8A16 (CUTLASS):    int8_mode=1
W8A8 (SmoothQuant): int8_mode=2
W6Ax (FlexQ):       int8_mode=5
Additionally, for multi-GPU testing, you need to modify the tensor_para_size parameter (set it to the number of GPUs).
```

Run e2e efficiency evaluation:
```
cd build_release

# For single-GPU LLaMA model evaluation
./bin/llama_example

# For single-GPU OPT model evaluation
./bin/multi_gpu_gpt_example

# For multi-GPU evaluation
mpirun -n 2 ./bin/llama_example
```

## Citation
This repository extends the work done in the FlexQ paper:
```
@article{zhang2025flexq,
  title={FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design},
  author={Zhang, Hao and Jia, Aining and Bu, Weifeng and Cai, Yushu and Sheng, Kai and Chen, Hao and He, Xin},
  journal={arXiv preprint arXiv:2508.04405},
  year={2025}
}
```
