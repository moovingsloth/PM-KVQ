"""Versioned, strictly validated calibration metadata and runtime settings."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ThinKVConfig:
    thresholds: tuple = (0.3, 0.7)
    selected_layers: tuple = (0, 1, 2, 3)
    token_budget: int = 1024
    refresh_interval: int = 128
    reasoning_bits: int = 4
    execution_bits: int = 4
    transition_bits: int = 2
    group_size: int = 16
    quantization: bool = True
    eviction: bool = True

    def __post_init__(self):
        if len(self.thresholds) != 2 or not all(
            type(t) in (int, float) and math.isfinite(t) for t in self.thresholds
        ) or not 0 <= self.thresholds[0] < self.thresholds[1] <= 1:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
        if not self.selected_layers or any(type(i) is not int or i < 0 for i in self.selected_layers):
            raise ValueError("selected_layers must contain nonnegative layer indices")
        if list(self.selected_layers) != sorted(set(self.selected_layers)):
            raise ValueError("selected_layers must be sorted and unique")
        for name in ("token_budget", "refresh_interval", "group_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_size != 16:
            raise ValueError("ThinKV reference requires group_size=16")
        if type(self.reasoning_bits) is not int or self.reasoning_bits not in (4, 8):
            raise ValueError("reasoning_bits must be 4 or 8")
        if type(self.execution_bits) is not int or self.execution_bits != 4:
            raise ValueError("execution_bits must be 4 (NVFP4)")
        if type(self.transition_bits) is not int or self.transition_bits != 2:
            raise ValueError("transition_bits must be 2 (ternary)")
        if type(self.quantization) is not bool or type(self.eviction) is not bool:
            raise ValueError("compression switches must be boolean")

    def classify(self, sparsity):
        if not math.isfinite(sparsity) or not 0 <= sparsity <= 1:
            raise ValueError("sparsity must be finite and in [0, 1]")
        if sparsity < self.thresholds[0]:
            return "execution"
        if sparsity < self.thresholds[1]:
            return "reasoning"
        return "transition"

    def bits(self, thought):
        return getattr(self, f"{thought}_bits")


def model_identity(config, path):
    architecture = config.to_dict()
    for key in list(architecture):
        if key.startswith("_") or key in ("transformers_version", "torch_dtype"):
            architecture.pop(key)
    digest = hashlib.sha256(json.dumps(architecture, sort_keys=True).encode()).hexdigest()
    return {"path": str(Path(path).resolve()), "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers, "config_sha256": digest}


def validate_calibration(data, identity=None):
    required = {"schema_version", "model", "dataset", "generation", "seed", "numerical",
                "thresholds", "selected_layers", "kde", "diagnostics", "versions"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError(f"calibration keys must be exactly {sorted(required)}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported ThinKV calibration schema")
    numerical = data["numerical"]
    expected = {"group_size", "reasoning_bits", "execution_bits", "transition_bits", "refresh_interval"}
    if not isinstance(numerical, dict) or set(numerical) != expected:
        raise ValueError("invalid numerical settings")
    if not isinstance(data["thresholds"], list) or not isinstance(data["selected_layers"], list):
        raise ValueError("thresholds and selected_layers must be JSON arrays")
    config = ThinKVConfig(thresholds=tuple(data["thresholds"]),
                         selected_layers=tuple(data["selected_layers"]), **numerical)
    model = data["model"]
    if not isinstance(model, dict) or set(model) != {"path", "model_type", "num_hidden_layers", "config_sha256"}:
        raise ValueError("invalid model identity")
    if model["model_type"] not in ("qwen2", "llama") or not isinstance(model["path"], str) or not model["path"]:
        raise ValueError("invalid model path/type")
    digest = model["config_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid model config digest")
    if type(model["num_hidden_layers"]) is not int or len(config.selected_layers) != 4 or max(config.selected_layers) >= model["num_hidden_layers"]:
        raise ValueError("calibration must select four valid model layers")
    if identity is not None and model != identity:
        raise ValueError("ThinKV calibration model identity does not match the loaded model")
    dataset = data["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {"path", "fingerprint", "prompt_field", "indices", "prompt_sha256"}:
        raise ValueError("invalid dataset provenance")
    if any(not isinstance(dataset[k], str) or not dataset[k] for k in ("path", "fingerprint", "prompt_field", "prompt_sha256")):
        raise ValueError("dataset provenance strings must be nonempty")
    if len(dataset["prompt_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in dataset["prompt_sha256"]):
        raise ValueError("invalid prompt content digest")
    indices = dataset["indices"]
    if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("dataset indices must be nonempty, unique nonnegative integers")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    generation = data["generation"]
    if not isinstance(generation, dict) or set(generation) != {"max_new_tokens", "do_sample", "temperature", "top_p"}:
        raise ValueError("invalid generation settings")
    if type(generation["max_new_tokens"]) is not int or generation["max_new_tokens"] < 2 or generation["do_sample"] is not True:
        raise ValueError("calibration needs sampled generation with at least two tokens")
    for key, upper in (("temperature", math.inf), ("top_p", 1)):
        value = generation[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"invalid generation {key}")
    if data["kde"] != {"bandwidth": "scott", "grid_size": 4096, "layer_rule": "three_modes_every_prompt_first_four"}:
        raise ValueError("unsupported calibration KDE conventions")
    if not isinstance(data["diagnostics"], dict) or not data["diagnostics"]:
        raise ValueError("calibration diagnostics are required")
    for layer in config.selected_layers:
        diagnostic = data["diagnostics"].get(str(layer))
        if not isinstance(diagnostic, dict) or set(diagnostic) != {"mode_counts", "errors"}:
            raise ValueError("selected layer diagnostics are required")
        counts = diagnostic["mode_counts"]
        if not isinstance(counts, list) or len(counts) != len(indices) or any(type(c) is not int or c != 3 for c in counts):
            raise ValueError("selected layers must have three modes on every calibration prompt")
        if diagnostic["errors"] != [None] * len(indices):
            raise ValueError("selected layers must have successful calibration diagnostics")
    if not isinstance(data["versions"], dict) or set(data["versions"]) != {"torch", "transformers", "scipy", "python"} or any(not isinstance(v, str) or not v for v in data["versions"].values()):
        raise ValueError("calibration software versions are required")
    # Reject NaN/Infinity anywhere, including diagnostics.
    json.dumps(data, allow_nan=False)
    return config


def load_calibration(path, identity=None):
    with open(path) as handle:
        data = json.load(handle)
    return data, validate_calibration(data, identity)


def numerical_settings(config):
    return {k: v for k, v in asdict(config).items() if k in
            ("group_size", "reasoning_bits", "execution_bits", "transition_bits", "refresh_interval")}
