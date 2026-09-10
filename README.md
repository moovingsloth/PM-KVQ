# PM-KVQ: R1-Qwen-14B on AIME2025

Personal experiment workspace for **Progressive Mixed-precision KV Cache
Quantization for Long-CoT LLMs (PM-KVQ)**. PM-KVQ gradually lowers KV-cache
precision, allocates memory by layer sensitivity, and uses positional
interpolation during calibration. Upstream authors and maintainers are credited
below.

This guide runs **DeepSeek-R1-Distill-Qwen-14B** on **AIME2025**, comparing
BF16 (`original`) with PM-KVQ (`fake`) on the local NVIDIA DGX Spark (GB10,
aarch64). The fake backend measures quantization's numerical effects; it does
**not** physically compress the KV tensors. These commands cover one model and
benchmark, not the entire paper's experiment suite.

## 1. Set up the environment

For an existing `pm_kvq` environment, activate it and run the check below.
For a fresh environment on this GB10 host:

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda create -n pm_kvq python=3.10 -y
conda activate pm_kvq
python -m pip install -r requirements.txt
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e . --no-deps
```

The requirements pin PyTorch 2.5.1; the local GB10 setup uses 2.9.1 with CUDA
13.0 instead. Do not reinstall the requirements afterward and undo that override.
Keep Transformers 4.51.3 and the paired Tokenizers constraint: the model and
sampler hooks depend on those APIs. This CUDA override is specific to GB10;
it is not a general installation command for other GPUs.

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); x = torch.randn(128, 128, device="cuda"); print((x @ x).mean().item())'
python scripts/smoke_aime2025.py --help
```

The CUDA check must report availability and complete the matrix multiplication.
A successful check alone does not validate a full experiment.

## 2. Check local inputs

All run commands below use these existing locations. Change `--model_path` and
`--dataset_root` if your assets live elsewhere.

| Input | Required location |
| --- | --- |
| Model config, tokenizer, and weight shards | `/home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B/` |
| AIME2025 I and II | `/home/dongwon/workspace/datasets/aime/aime_2025_{I,II}/` |
| RedPajama calibration dataset | `/home/dongwon/workspace/datasets/redpajama-1t-sample/` |

The dataset root must contain both `aime/` and `redpajama-1t-sample/`.
AIME subsets contain `problems/1.tex` and the remaining benchmark files.
The calibration loader expects a Hugging Face dataset with a `train` split and
`text` and `meta` columns. Check the active shared dataset mount at
`/home/dongwon/mnt/seraph-datasets` before downloading or copying data; reuse
existing assets and keep that mount read-only.

The runner checks CUDA, model shard presence, and the first AIME problem in each
subset before calibration. These checks do not establish complete dataset
integrity. Hugging Face caches are automatically placed in this repository's
`.cache/huggingface/`.

## 3. Choose and run an experiment

Each preset runs sensitivity profiling → memory allocation → key calibration →
scale search → BF16 evaluation and judging → PM-KVQ evaluation and judging.
Generation is sequential on `main`.

| Preset | Calibration samples | Problems × responses per problem | Responses per method | Purpose |
| --- | ---: | --- | ---: | --- |
| `smoke` | 8 | 2 × 1 (I.1 and II.1) | 2 | Check the complete pipeline |
| `day` | 8 | 10 × 4 (I.1–5 and II.1–5) | 40 | Reduced experiment |
| `full` | 512 | 30 × 16 (all I and II) | 480 | Full selected benchmark |

Smoke and day results are not paper benchmark scores. Full runs generate
**960 responses total** across the two methods and can take substantial time;
use a persistent terminal session.

### Smoke

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
python scripts/smoke_aime2025.py \
  --preset smoke \
  --model_path /home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B \
  --dataset_root /home/dongwon/workspace/datasets \
  --wandb_entity dlehddnjs245-kyung-hee-university \
  --wandb_project pm-kvq
```

### Day

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
python scripts/smoke_aime2025.py \
  --preset day \
  --model_path /home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B \
  --dataset_root /home/dongwon/workspace/datasets \
  --wandb_entity dlehddnjs245-kyung-hee-university \
  --wandb_project pm-kvq
```

### Full: R1-Qwen-14B / AIME2025 / BF16 vs PM-KVQ

```bash
cd /home/dongwon/workspace/emil/pm-pkv
conda activate pm_kvq
python scripts/smoke_aime2025.py \
  --preset full \
  --model_path /home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B \
  --dataset_root /home/dongwon/workspace/datasets \
  --wandb_entity dlehddnjs245-kyung-hee-university \
  --wandb_project pm-kvq
```

Shared settings are 2,048-token calibration sequences, effective length 8,192,
a **1,024 MiB per-request KV budget**, mixed layer choices `{2,4}`, and 2-bit
key/value scale calibration. Generation uses BF16 weights, temperature 0.6,
top-p 0.95, and a 32,768-token output limit. Seeds start at 42 (42–57 for full).
The initial cache, one sink token, and 128-token window use 16 bits.
Do not increase the KV budget to fill Spark's memory: that changes the
quantization schedule and the experiment.

## 4. Inspect results and logging

Outputs go to `outputs/<preset>/<timestamp>/`. Add `--output_dir /path/to/new/run`
to choose a directory; it must not already exist.

| Artifact | What to inspect |
| --- | --- |
| `metadata.json` | Settings, versions, Git revision, status, stage commands, exit codes, and elapsed times |
| `*.log` | Individual stage output and judge metrics |
| `sensitivity.pt`, `budgets.pt`, `max_keys.pt`, `scales.pt` | Calibration and allocation artifacts |
| `original/*.json`, `pm-kvq/*.json` | Responses and judgements; full uses `responses.json` in each directory |
| `summary.json` | Correct/total counts, token-limit hits, output lengths, and whether quantization occurred |

Response files contain JSON objects, not JSONL. Keep only response files in each
method's directory. The judge reports `incomplete test data:2/480` for smoke and
`40/480` for day; those are expected subset sizes. A short response may finish
before a quantization transition, so inspect `quantization_observed`.

W&B is **enabled by default**, using the entity and project shown in the commands.
Authenticate with your existing W&B setup before running, or append `--no-wandb`
to any run command. Add `--wandb_name` to set a run name. Stage output and metrics
are logged during execution; calibration figures are also uploaded.

Failures stop the pipeline and retain artifacts. Inspect `metadata.json` and the
failed stage log before retrying. A rerun creates a fresh directory and **does
not resume partial responses**. Completion and coverage checks do not prove
physical memory savings or reproduce the entire paper.

## Advanced use

Use the individual scripts for custom calibration and evaluation:
`get_sensitivity.py`, `allocate_memory.py`, `get_max_keys.py`,
`search_rep_scales.py`, `evaluation.py`, and `judge.py` under `scripts/`.
Run each with `--help` for its arguments. RotateKV additionally requires
`fast-hadamard-transform`; it is not needed for this BF16/PM-KVQ workflow.

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
