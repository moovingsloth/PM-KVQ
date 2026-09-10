import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.quantization.methods.quant_wrapper import quantize_model
from pm_kvq.evaluation.eval_wrapper import evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the model")
parser.add_argument("--dataset_path", type=str, default=None, help="Local benchmark dataset directory")
parser.add_argument("--aime_manifest", default=None, help="Ordered AIME selection JSON; overrides version/start/end")
parser.add_argument("--output_path", type=str, help="Path to the output .jsonl file")
parser.add_argument("--benchmark", type=str, help="Benchmark name", default="aime", choices=["aime", "cmimc", "livecodebench"])
parser.add_argument("--version", type=str, help="Benchmark version", default="2024")
parser.add_argument("--seed", type=int, help="Random seed for the first response", default=42)
parser.add_argument("--start", type=int, help="Start problem index", default=0)
parser.add_argument("--end", type=int, help="End problem index", default=30)
parser.add_argument("--n_responses", type=int, help="Number of responses per problem", default=16)
parser.add_argument("--method", type=str, help="KV cache method", default="original", choices=["original", "pm-kvq", "rtn", "kivi", "rotatekv", "mikv", "thinkv"])
# Register before the first parse so --method thinkv --help includes its options.
parser.add_argument("--thinkv_calibration", help="Validated ThinKV calibration JSON")
parser.add_argument("--thinkv_token_budget", type=int, default=1024)
parser.add_argument("--thinkv_refresh_interval", type=int, default=None)
parser.add_argument("--thinkv_reasoning_bits", type=int, choices=[4, 8], default=None)
parser.add_argument("--thinkv_execution_bits", type=int, choices=[4], default=None)
parser.add_argument("--thinkv_transition_bits", type=int, choices=[2], default=None)
args, unknown = parser.parse_known_args()

if args.method == "original":
    pass

elif args.method == "thinkv":
    pass

elif args.method == "pm-kvq":
    parser.add_argument("--backend", help="Backend to implement PM-KVQ", type=str, default="fake", choices=["fake", "real"])
    parser.add_argument("--rep_scales", help="Path to reparameterization scales", type=str, default=None)
    def budget_value(value):
        try:
            return float(value)
        except ValueError:
            return value
    parser.add_argument("--kv_budgets", help="Budget artifact path or per-layer MB", type=budget_value, required=True)
    parser.add_argument("--n_sink_token", help="Number of sink tokens", type=int, default=1)
    parser.add_argument("--n_sink_token_bits", help="Bit-width of sink tokens", type=int, default=16)
    parser.add_argument("--n_window_token", help="Number of tokens in sliding window", type=int, default=128)
    parser.add_argument("--n_window_token_bits", help="Bit-width of tokens in sliding window", type=int, default=16)
    parser.add_argument("--n_init_kv_bits", help="Initial bit-width of KV Cache", type=int, default=16)

elif args.method == "rtn":
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

elif args.method == "kivi":
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

elif args.method == "rotatekv":
    parser.add_argument("--k_reorder_indices", help="Path to reorder indices", type=str, default=None)
    parser.add_argument("--k_had_dim", help="Dimension of head-wise hadamard, default to -1", type=int, default=-1)
    parser.add_argument("--n_pivot_token", help="Number of pivot tokens", type=int, default=20)
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

elif args.method == "mikv":
    parser.add_argument("--n_sink_tokens", help="Number of sink tokens", type=int, default=0)
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

else:
    raise NotImplementedError

args = parser.parse_args()
if args.method == "thinkv" and not args.thinkv_calibration:
    parser.error("--method thinkv requires --thinkv_calibration")
if args.n_responses <= 0:
    parser.error("--n_responses must be positive")
if args.aime_manifest is not None:
    if args.benchmark != "aime":
        parser.error("--aime_manifest requires --benchmark aime")
    from pm_kvq.evaluation.aime_manifest import load_manifest, load_problems
    from pm_kvq.evaluation.eval_aime import DEFAULT_DATASET_PATH
    # Fail on missing selected data before from_pretrained can initialize a GPU.
    load_problems(args.dataset_path or DEFAULT_DATASET_PATH, load_manifest(args.aime_manifest))
args_dict = vars(args)
if args.method != "thinkv":
    args_dict = {key: value for key, value in args_dict.items() if not key.startswith("thinkv_")}
dataset_path = args_dict.pop("dataset_path")
aime_manifest = args_dict.pop("aime_manifest")
method_kwargs = {key: args_dict[key] for key in args_dict if key not in ["model_path", "output_path", "seed", "benchmark", "version", "start", "end", "n_responses", "method"]}
evaluate_kwargs = {key: args_dict[key] for key in args_dict if key in ["output_path", "seed", "version", "start", "end", "n_responses"]}
generation_kwargs = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 32768, "do_sample": True}
if dataset_path is not None:
    evaluate_kwargs["dataset_path"] = dataset_path
if aime_manifest is not None:
    evaluate_kwargs["aime_manifest"] = aime_manifest

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

quantize_model(model, args.method, method_kwargs)
evaluate_model(model, tokenizer, args.benchmark, evaluate_kwargs, generation_kwargs)
