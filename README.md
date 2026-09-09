# PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

## Abstract

Recently, significant progress has been made in developing reasoning-capable Large Language Models (LLMs) through long Chain-of-Thought (CoT) techniques. However, this long-CoT reasoning process imposes substantial memory overhead due to the large Key-Value (KV) Cache memory overhead. Post-training KV Cache quantization has emerged as a promising compression technique and has been extensively studied in short-context scenarios. However, directly applying existing methods to long-CoT LLMs causes significant performance degradation due to the following two reasons: (1) **Large cumulative error**: Existing methods fail to adequately leverage available memory, and they directly quantize the KV Cache during each decoding step, leading to large cumulative quantization error. (2) **Short-context calibration**: Due to Rotary Positional Embedding (RoPE), the use of short-context data during calibration fails to account for the distribution of less frequent channels in the Key Cache, resulting in performance loss. We propose **P**rogressive **M**ixed-Precision **KV** Cache **Q**uantization (**PM-KVQ**) for long-CoT LLMs to address the above issues in two folds: (1) To reduce cumulative error, we design a progressive quantization strategy to gradually lower the bit-width of KV Cache in each block. Then, we propose block-wise memory allocation to assign a higher bit-width to more sensitive transformer blocks. (2) To increase the calibration length without additional overhead, we propose a new calibration strategy with positional interpolation that leverages short calibration data with positional interpolation to approximate the data distribution of long-context data. Extensive experiments on 7B–70B long-CoT LLMs show that PM-KVQ improves reasoning benchmark performance by up to 8% over SOTA baselines under the same memory budget.

## Installation

1. Create a new conda environment.

   ```bash
   conda create -n pm_kvq python==3.10
   conda activate pm_kvq
   ```

2. Use pip to install packages from requirements.

   ```bash
   pip install -r requirements.txt
   ```

3. Install `pm_kvq` from source.

   ```bash
   pip install -e .
   ```

4. For RotateKV baseline, install `fast-hadamard-transform` from [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform).

## NVIDIA DGX Spark (GB10): R1-Qwen-14B on AIME2025

These commands and timings are for the local **NVIDIA DGX Spark** (GB10
Superchip, aarch64, 121.6 GiB unified LPDDR5x, driver CUDA 13.0). They are not
A100/H100 numbers. Use the existing Conda environment; `uv` is not required.
Run from this repository:

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
export HF_HOME=/home/dongwon/workspace/emil/pm-pkv/.cache/huggingface
export HF_XET_CACHE="$HF_HOME/xet"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); x = torch.randn(128, 128, device="cuda"); print((x @ x).mean().item())'
```

`requirements.txt` pins `torch==2.5.1`. On this ARM64 DGX Spark (GB10) that pin
installs a CPU-only wheel (`2.5.1 False`), so the check raises
`AssertionError: Torch not compiled with CUDA enabled`. The driver reports
CUDA 13.0; there is no aarch64 CUDA wheel for 2.5.1.

Install the CUDA 13.0 build below in the same Conda environment. Leave
`transformers==4.51.3` unchanged; `torch==2.9.1` does not require a newer
Transformers. This repo copies 4.51-era Qwen2/Llama attention and cache
internals and monkey-patches `model._sample`. That copy was already adjusted
for 4.51.3's `_has_unfinished_sequences(this_peer_finished, synced_gpus, device)`
signature (the 4.49 form also passed `cur_len` and `max_length`). A further
Transformers bump would break those hooks and the paired
`tokenizers>=0.21,<0.22` pin. Reinstall `pm_kvq` with `--no-deps`. Do not
reinstall `requirements.txt` afterward, because that restores the CPU-only
PyTorch pin.

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e . --no-deps
```

The check then prints `2.9.1+cu130 True` and a matmul mean, on device
`NVIDIA GB10`. PyTorch 2.9.1+cu130 ships kernels through sm_120 plus PTX;
GB10 is sm_121, so CUDA init warns that the GPU is outside the wheel's
(8.0)–(12.0) range. The matmul still runs.

Local reusable assets (not the network mount):

| Asset | Location |
| --- | --- |
| AIME2025 I and II | `/home/dongwon/workspace/datasets/aime/` |
| RedPajama calibration source | `/home/dongwon/workspace/datasets/redpajama-1t-sample/` |
| Selected model | `/home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B/` |
| Run artifacts and logs | `/home/dongwon/workspace/emil/pm-pkv/outputs/smoke/` |

The cache exports override any inherited mount-based Hugging Face configuration.
The smoke runner sets these same local cache paths automatically.
The datasets were copied from the existing local repository assets. To obtain
the selected model at the path above (also resumes an incomplete download):

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", local_dir="/home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B")'
```

Run the smoke pipeline with the Conda interpreter that has the CUDA 13.0
PyTorch build. The runner sets `HF_HOME` and related cache paths itself.
**W&B logging is on by default** and streams the Logs tab online to
[pm-kvq](https://wandb.ai/dlehddnjs245-kyung-hee-university/pm-kvq).

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
export WANDB_ENTITY=dlehddnjs245-kyung-hee-university
export WANDB_PROJECT=pm-kvq
/home/dongwon/miniconda3/envs/pm_kvq/bin/python scripts/smoke_aime2025.py \
  --model_path /home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B \
  --dataset_root /home/dongwon/workspace/datasets \
  --wandb \
  --wandb_entity dlehddnjs245-kyung-hee-university \
  --wandb_project pm-kvq
```

Optional: `--output_dir /path/to/run` writes artifacts to a chosen directory
instead of `outputs/smoke/<YYYYMMDD-HHMMSS>/`. The directory must not already
exist. A completed example is `outputs/smoke/20260909-151318`.

Credentials come from `WANDB_API_KEY`, `~/.env`, or an existing `wandb login`
(`~/.netrc`). Stage stdout is captured into the run Logs tab while the pipeline
is still running; tqdm carriage returns are collapsed to the last line. Each run
also logs paper Fig. 3/4: per-block KV quantization sensitivity and block-wise
memory allocation, with one color per total KV budget. Pass `--no-wandb` to skip
logging. Override the run name with `--wandb_name` or `WANDB_NAME`. To publish
an already-finished output directory without rerunning:

```bash
/home/dongwon/miniconda3/envs/pm_kvq/bin/python scripts/smoke_aime2025.py \
  --log_existing outputs/smoke/20260909-151318 \
  --wandb_entity dlehddnjs245-kyung-hee-university \
  --wandb_project pm-kvq
```

This runs sensitivity profiling, memory allocation, key calibration, scale
search, then BF16 and PM-KVQ evaluation of problem 1 from each AIME2025 subset
(indices 0 and 15), with one response per problem. It creates a new timestamped
output directory and records each command, exit code, elapsed time and response.
Failure stops the pipeline; inspect the named stage log before rerunning.

Settings follow the [original paper](https://arxiv.org/html/2505.18610v1):
2,048-token calibration sequences, effective length 8,192, 20-point scale search,
2-bit scale calibration, layer choices `{2,4}`, and the Qwen-14B mixed-precision
budget ratio of 16/12 relative to the 2-bit budget at 32,768 tokens (1,024 MiB per
request). Generation uses temperature 0.6, top-p 0.95, seed 42 and a 32,768-token
output limit. Calibration uses **8 samples instead of 512**, and evaluation uses
**2 problems × 1 response instead of 30 × 16**. These reductions make this a
pipeline smoke test, not a reproduction of the reported benchmark score.

Paper Table 2 batch sizes for Qwen-14B are a **target-GPU memory scenario**,
not the evaluation loop. On `1×A100-40G` the 14B BF16 weights leave about 12 GiB
for a 32,768-token KV cache, which fills at **BS=16** with uniform 2-bit
(768 MiB/request) or **BS=12** with mixed 2/4 (1,024 MiB/request). Uniform 4-bit
would need 1,536 MiB/request and does not fit that leftover. Accuracy in that
table is fake-quantized; the 12/16 figures set the per-request budget, not a
requirement to call `generate` with a batch of 12.

This host is an **NVIDIA DGX Spark** (GB10 Superchip) with 121.6 GiB unified
memory. The 14B weights are 27.5 GiB, leaving about 94 GiB. Matching the
paper's *bit-width* row still means 1,024 MiB/request (BS=12 mixed) or
768 MiB/request (BS=16 2-bit). The runner uses 1,024 MiB. Do not raise that
budget just because Spark is larger; a larger budget changes the quantization
schedule and is no longer the paper's 2/4 row.

If Spark is treated as the target GPU and leftover memory is filled at 32,768
tokens (activations and OS not counted):

| KV storage | Setup | Per-request KV @32k | Batch that fills ~94 GiB |
| --- | --- | --- | --- |
| fake (current smoke) | tensors stay BF16-sized | 6,144 MiB | ~15 |
| packed `real` | mixed 2/4 (paper BS=12 budget) | 1,024 MiB | ~94 |
| packed `real` | uniform 2-bit (paper BS=16 budget) | 768 MiB | ~125 |
| packed `real` | uniform 4-bit | 1,536 MiB | ~62 |
| original BF16 | 16-bit KV | 6,144 MiB | ~15 |

So on this GPU, fake 32k serving is about **BS=12–16**, coincidentally the
paper's A100-40G *quantized* batch, because fake does not shrink the tensors.
Packed `real` could go much higher at the same 1,024 MiB/request. The fake
backend simulates quantization numerically; its allocations are not evidence
of physically compressed GPU memory.

On `main`, generation is still **one sequence at a time** and each calibration
stage reloads the 14B weights (the in-flight day run uses this loop). Branch
`spark` batches BF16 (`original`) responses (cap 4) and loads calibration
weights once. PM-KVQ stays sequential on both: progressive `n_bits`, sink,
window, and `n_kv_bits` are per-layer scalars, not per-sequence. Batching that
path would share one quantization schedule across different CoT lengths.

### Runtime on DGX Spark

Measured on this box, smoke run `outputs/smoke/20260909-151318` (`main`,
sequential, three separate calibration processes):

| Stage | Wall time | What the GPU was doing |
| --- | --- | --- |
| sensitivity | 223 s | ~180 s loading 4 shards (~55 s/shard), then 8×3.5 s compute |
| max_keys | 197 s | loads the 14B weights again |
| scales | 340 s | third load; `search_rep_scales.py` had no `device_map` |
| original generate | **7.1 tok/s** | 14B BF16 decode ≈ 196 GB/s of ~273 GB/s LPDDR5x |
| PM-KVQ fake generate | **6.0–6.4 tok/s** | same decode plus per-token fake-quant |

`nvidia-smi` at **1% GPU** during `Loading checkpoint shards` is expected: that
phase is host memcpy of safetensors, not matmuls. After load, batch-1 decode is
memory-bandwidth bound, so SM util% still looks modest while the bus is nearly
full. Day preset is 80 sequential generates; calibration (~12 min) is not the
limiter.

The `spark` branch (worktree `../pm-pkv-spark`) applies host-side speedups that
keep the paper 1,024 MiB budget:

1. Load the 14B weights **once** for max-keys, scale search, sensitivity, and
   allocation (`scripts/calibrate_pmkvq.py`).
2. Run scale search on GPU (`device_map="auto"`) and skip per-layer CPU offload
   when the model already fits in Spark unified memory.
3. Evaluate all selected AIME slices in **one process per method**, so the
   weights are not reloaded between I and II.
4. Batch **original** responses up to 4. Decode rereads 28 GiB of weights per
   step regardless of batch, so batch 4 amortizes that traffic.
5. Leave **PM-KVQ sequential** (see above).

Do not check out `spark` over an in-flight `main` run: later stages spawn
scripts from the working tree. Weight-only quantization, vLLM, and raising the
KV budget are out of scope: they change the experiment, not just the host
mapping. PyTorch 2.9.1+cu130 still warns that GB10 (sm_121) is outside the
wheel's (8.0)–(12.0) range; kernels run via sm_120 PTX.

The original judge prints `incomplete test data:2/480` for each method; that is
expected for this subset. Responses are JSON objects despite older instructions
calling them JSONL. Keep only response files in each method's response directory.
Each response records elapsed time and token-limit status; PM-KVQ responses also
record the cache bit counts for each layer. `summary.json` reports whether actual
quantization occurred. A short response may finish before the first transition.

The smoke workflow includes compatibility fixes for the pinned Transformers
sampler API, budget-file CLI arguments, direct application of calibrated scales,
and safe loading of newly generated budget artifacts with modern PyTorch.
KV-budget exhaustion now raises an error instead of silently scoring an
unfinished generation.

## Day-scale experiment: 10 problems × 4 responses

Same runner and W&B defaults as the smoke test. Calibration stays at **8 samples**.
Evaluation uses **5 problems from AIME2025 I and 5 from II** (indices 0–4 and
15–19) with **4 responses each**, so 40 responses per method (80 total). On this
DGX Spark, `main` sequential smoke-length CoT is about **half a day to one day**.
Rerun from the `spark` branch to skip repeated 14B loads and batch the four
original responses; PM-KVQ is still sequential. This is not a paper score.

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
export WANDB_ENTITY=dlehddnjs245-kyung-hee-university
export WANDB_PROJECT=pm-kvq
/home/dongwon/miniconda3/envs/pm_kvq/bin/python scripts/smoke_aime2025.py \
  --preset day \
  --model_path /home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B \
  --dataset_root /home/dongwon/workspace/datasets \
  --wandb \
  --wandb_entity dlehddnjs245-kyung-hee-university \
  --wandb_project pm-kvq
```

Artifacts go to `outputs/day/<timestamp>/`. Seeds are 42–45 per problem. The
judge will print `incomplete test data:40/480`. Pass `--no-wandb` to skip
logging.

## Full experiment: R1-Qwen-14B on AIME2025

After the DGX Spark environment setup above, run the same pipeline runner with
`--preset full`. It uses **512 calibration samples** and evaluates **all 30
problems × 16 responses = 480 responses per method**, comparing BF16
(`original`) with PM-KVQ (`fake`). **W&B logging is on by default** and streams
to [pm-kvq](https://wandb.ai/dlehddnjs245-kyung-hee-university/pm-kvq). This
covers the selected model on AIME2025, not the entire paper's experiment suite.

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
export WANDB_ENTITY=dlehddnjs245-kyung-hee-university
export WANDB_PROJECT=pm-kvq
/home/dongwon/miniconda3/envs/pm_kvq/bin/python scripts/smoke_aime2025.py \
  --preset full \
  --model_path /home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B \
  --dataset_root /home/dongwon/workspace/datasets \
  --wandb \
  --wandb_entity dlehddnjs245-kyung-hee-university \
  --wandb_project pm-kvq
```

The evaluation uses BF16 weights, temperature 0.6, top-p 0.95, a 32,768-token
output limit, and seeds 42–57 for each problem. Generation runs sequentially
(960 responses total). PM-KVQ uses the 1,024 MiB per-request budget and mixed
`{2,4}` settings; `fake` measures numerical accuracy without physical KV
compression. Calibration artifacts are regenerated at 512 samples. Pass
`--no-wandb` to skip logging.

Artifacts go to `outputs/full/<timestamp>/`. Responses are JSON objects in
`{original,pm-kvq}/responses.json`. `summary.json`, stage logs, and the W&B
Logs tab record pass@1, voting accuracy, and per-stage stdout. Failures stop
the pipeline and retain artifacts. Rerunning creates a fresh output directory;
it does not resume partial responses. Use a long-lived terminal session for the
full run.

## Apply PM-KVQ

### Block-wise Memory Allocation

1. Profile the sensitivity to quantization of KV Cache in different transformer blocks.

   ```bash
   python scripts/get_sensitivity.py \
   --model_path /PATH/TO/MODEL \
   --dataset_path /PATH/TO/CALIBRATION/DATASET \
   --n_samples 512 \
   --seq_len 2048 \
   --effective_len 8192 \
   --save_path /PATH/TO/SAVE/SENSITIVITY
   ```

2. Assign memory budget to each transformer block. The value of `--memory_budget` is specified in megabytes (MB).

   ```bash
   python scripts/allocate_memory.py \
   --sensitivity_path /PATH/TO/SENSITIVITY \
   --memory_budget 1024 \
   --fbit_choices 4,2 \
   --hidden_size ${HIDDEN_DIMENSION_OF_MODEL} \
   --max_len 32768 \
   --save_path /PATH/TO/SAVE/MEMORY/BUDGET
   ```

### Calibration with Positional Interpolation

1. Calculate maximum magnitude of the Key cache.

   ```bash
   python scripts/get_max_keys.py \
   --model_path /PATH/TO/MODEL \
   --dataset_path /PATH/TO/CALIBRATION/DATASET \
   --n_samples 512 \
   --seq_len 2048 \
   --effective_len 8192 \
   --save_path /PATH/TO/SAVE/MAX/KEYS
   ```

2. Search for the optimal reparameterization factor.

   ```bash
   python scripts/search_rep_scales.py \
   --model_path /PATH/TO/MODEL \
   --dataset_path /PATH/TO/CALIBRATION/DATASET \
   --n_samples 512 \
   --seq_len 2048 \
   --effective_len 8192 \
   --max_keys_path /PATH/TO/MAX/KEYS \
   --k_bits 4 \
   --v_bits 4 \
   --save_path /PATH/TO/SAVE/REP/SCALES
   ```

### Quantization and Evaluation

1. Evaluate the quantized model and save its responses to a `.jsonl` file. Use the `--start` and `--end` options to specify the range of problem indices to evaluate. To facilitate joint judgement, save the response files for different problems in the same directory.

   ```bash
   python scripts/evaluation.py \
   --model_path /PATH/TO/MODEL \
   --output_path /PATH/TO/SAVE/MODEL/RESPONSES \
   --benchmark aime \
   --version 2024 \
   --start 0 \
   --end 30 \
   --n_responses 16 \
   --method pm-kvq \
   --backend fake \
   --rep_scales /PATH/TO/REP/SCALES \
   --kv_budgets /PATH/TO/MEMORY/BUDGET \
   --n_sink_token 1 \
   --n_sink_token_bits 16 \
   --n_window_token 128 \
   --n_window_token_bits 16 \
   --n_init_kv_bits 16
   ```

2. Judge the responses and calculate the evaluation metrics.

   ```bash
   python scripts/judge.py \
   --benchmark aime \
   --version 2024 \
   --responses_dir /PATH/TO/MODEL/RESPONSES
   ```

## Contact us

- Tengxuan Liu: [liutx21@mails.tsinghua.edu.cn](mailto:liutx21@mails.tsinghua.edu.cn)
- Shiyao Li: [lishiyao20@mails.tsinghua.edu.cn](mailto:lishiyao20@mails.tsinghua.edu.cn)
- Jiayi Yang: [jy.yang1030@gmail.com](mailto:jy.yang1030@gmail.com)
- Yu Wang: [yu-wang@tsinghua.edu.cn](mailto:yu-wang@tsinghua.edu.cn)

This work is maintained by [NICS-EFC Lab](https://nicsefc.ee.tsinghua.edu.cn/) (Tsinghua University) and [Infinigence-AI](https://www.infini-ai.com/) (Beijing China).



<p align="middle">
  <img src="figures/logo_nicsefc.jpg" width="35%" hspace="30" />
  <img src="figures/logo_Infinigence-ai.png" width="35%" hspace="30" />
</p>
