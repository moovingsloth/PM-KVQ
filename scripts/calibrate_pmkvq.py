"""Load the 14B weights once and run PM-KVQ calibration (DGX Spark)."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.datasets.calib_dataset import get_calib_redpajama
from pm_kvq.quantization.methods.pm_kvq.allocation.allocation import allocate_memory_budget
from pm_kvq.quantization.methods.pm_kvq.allocation.sensitivity import get_kv_sensitivity
from pm_kvq.quantization.methods.pm_kvq.smoothattention.apply_smoothattention import get_max_keys
from pm_kvq.quantization.methods.pm_kvq.smoothattention.searching_scales import search_rep_scales

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--n_samples", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--effective_len", type=int, default=8192)
parser.add_argument("--sensitivity_path", type=str)
parser.add_argument("--max_keys_path", type=str)
parser.add_argument("--scales_path", type=str)
parser.add_argument("--k_bits", type=int, default=2)
parser.add_argument("--v_bits", type=int, default=2)
parser.add_argument("--memory_budget", type=float, default=None)
parser.add_argument("--fbit_choices", type=str, default="4,2")
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--max_len", type=int, default=32768)
parser.add_argument("--budgets_path", type=str, default=None)
args = parser.parse_args()

print("Loading model", flush=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
print("Tokenizing calibration data", flush=True)
calib_dataset = get_calib_redpajama(args.dataset_path, args.n_samples, args.seq_len, tokenizer)

# max_keys / scales before sensitivity: get_kv_sensitivity monkey-patches attention.
print("Profiling max keys", flush=True)
max_keys = get_max_keys(model, calib_dataset, args.effective_len, args.max_keys_path)

print("Searching reparameterization scales", flush=True)
search_rep_scales(
    model,
    k_config={"n_bits": args.k_bits, "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False},
    v_config={"n_bits": args.v_bits, "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False},
    dataset=calib_dataset,
    max_keys=max_keys,
    effective_len=args.effective_len,
    save_path=args.scales_path,
)

print("Profiling KV sensitivity", flush=True)
get_kv_sensitivity(model, calib_dataset, args.effective_len, args.sensitivity_path)

if args.budgets_path is not None:
    print("Allocating memory budgets", flush=True)
    sensitivity = torch.load(args.sensitivity_path, map_location="cpu")
    allocate_memory_budget(
        list(map(int, args.fbit_choices.split(","))),
        sensitivity["k_sensitivity"],
        sensitivity["v_sensitivity"],
        args.memory_budget,
        args.hidden_size,
        args.max_len,
        args.budgets_path,
    )
