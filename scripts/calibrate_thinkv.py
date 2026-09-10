"""Calibrate ThinKV from local s1K prompts; no automatic dataset/model downloads."""

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True, help="Local s1K HF dataset directory")
    parser.add_argument("--prompt_field", default="question")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--refresh_interval", type=int, default=128)
    parser.add_argument("--reasoning_bits", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if args.n_prompts <= 0 or args.seed < 0 or args.max_new_tokens < 2:
        parser.error("n_prompts must be positive, seed nonnegative, max_new_tokens >= 2")
    model_path, dataset_path = Path(args.model_path).resolve(), Path(args.dataset_path).resolve()
    if not model_path.is_dir() or not dataset_path.is_dir():
        parser.error("model_path and dataset_path must be existing local directories")
    output = Path(args.output_path)
    if output.exists():
        parser.error("output_path already exists; choose a fresh artifact path")

    import scipy
    import torch
    import transformers
    from datasets import load_dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pm_kvq.quantization.methods.thinkv import ThinKVConfig, apply_thinkv
    from pm_kvq.quantization.methods.thinkv.calibration import calibrate_traces
    from pm_kvq.quantization.methods.thinkv.config import model_identity, numerical_settings, validate_calibration
    from pm_kvq.utils.chatbot import chat

    if (dataset_path / "state.json").is_file() or (dataset_path / "dataset_dict.json").is_file():
        dataset = load_from_disk(str(dataset_path))
        if "train" in dataset:
            dataset = dataset["train"]
    else:
        dataset = load_dataset(str(dataset_path), split="train")
    if args.n_prompts > len(dataset) or args.prompt_field not in dataset.column_names:
        parser.error("s1K dataset lacks enough prompts or the requested prompt field")
    indices = random.Random(args.seed).sample(range(len(dataset)), args.n_prompts)
    prompts = [dataset[i][args.prompt_field] for i in indices]
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        parser.error("every selected prompt must be nonempty text")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                                               device_map="auto", torch_dtype=torch.bfloat16)
    identity = model_identity(model.config, str(model_path))
    settings = ThinKVConfig(selected_layers=tuple(range(model.config.num_hidden_layers)),
                            refresh_interval=args.refresh_interval, reasoning_bits=args.reasoning_bits,
                            quantization=False, eviction=False)
    apply_thinkv(model, config=settings, collect_traces=True)
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": True, "temperature": .6, "top_p": .95}
    traces = {layer: [] for layer in settings.selected_layers}
    for index, prompt in enumerate(prompts):
        torch.manual_seed(args.seed + index)
        chat(model, tokenizer, text=prompt, **generation)
        for layer in traces:
            traces[layer].append(model.thinkv_last_traces[layer])
        print(f"Calibrated prompt {index + 1}/{len(prompts)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = calibrate_traces(traces)
    except ValueError as error:
        diagnostics = output.with_suffix(".failure.json")
        diagnostics.write_text(json.dumps({"error": str(error), "model": identity,
                                          "indices": indices, "seed": args.seed}, indent=2))
        raise ValueError(f"{error}; saved diagnostics to {diagnostics}") from error
    artifact = dict(result, schema_version=1, model=identity, seed=args.seed,
                    numerical=numerical_settings(settings), generation=generation,
                    dataset={"path": str(dataset_path), "fingerprint": dataset._fingerprint,
                             "prompt_field": args.prompt_field, "indices": indices,
                             "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest()},
                    versions={"torch": torch.__version__, "transformers": transformers.__version__,
                              "scipy": scipy.__version__, "python": platform.python_version()})
    validate_calibration(artifact, identity)
    with output.open("x") as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
    print(f"Validated ThinKV calibration: {output}")


if __name__ == "__main__":
    main()
