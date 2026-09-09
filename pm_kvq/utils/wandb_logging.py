"""Weights & Biases helpers for PM-KVQ smoke and evaluation runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ENTITY = "dlehddnjs245-kyung-hee-university"
DEFAULT_PROJECT = "pm-kvq"
PROJECT_URL = f"https://wandb.ai/{DEFAULT_ENTITY}/{DEFAULT_PROJECT}"
SECRETS_FILE = Path.home() / ".env"
WANDB_CHILD_UNSET = (
    "WANDB_RUN_ID",
    "WANDB_SERVICE",
    "WANDB__SERVICE",
    "WANDB_INITED",
    "WANDB_SESSION_ID",
)
CONFIG_KEYS = (
    "model_path",
    "dataset_root",
    "torch",
    "transformers",
    "gpu",
    "calibration_samples",
    "seq_len",
    "effective_len",
    "memory_budget_mb",
    "problem_indices",
    "n_responses",
    "seed",
    "max_new_tokens",
    "temperature",
    "top_p",
    "backend",
    "git_revision",
    "model_revision",
    "status",
)
MODEL_CONFIG_KEYS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "vocab_size",
    "torch_dtype",
)
ARTIFACT_FILES = (
    "metadata.json",
    "summary.json",
    "budgets.pt",
    "sensitivity.pt",
    "scales.pt",
    "max_keys.pt",
)


def require_api_key(secrets_file: Path = SECRETS_FILE) -> str:
    """Use an environment/.env key, or an existing W&B netrc login."""
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key and secrets_file.is_file():
        for line in secrets_file.read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if candidate.startswith("export "):
                candidate = candidate[7:].lstrip()
            if candidate.startswith("WANDB_API_KEY="):
                key = candidate.split("=", 1)[1].strip().strip("'\"")
                break
    if not key:
        netrc_file = Path.home() / ".netrc"
        if secrets_file == SECRETS_FILE and netrc_file.is_file():
            netrc_text = netrc_file.read_text(encoding="utf-8")
            if "machine api.wandb.ai" in netrc_text or "machine wandb.ai" in netrc_text:
                return "netrc"
        raise RuntimeError(
            f"W&B credentials were not found in the environment, {secrets_file}, or ~/.netrc"
        )
    os.environ["WANDB_API_KEY"] = key
    return key


def child_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy the environment but detach child processes from the parent W&B run."""
    env = dict(os.environ if source is None else source)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    for key in WANDB_CHILD_UNSET:
        env.pop(key, None)
    return env


def wandb_config_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    config = {key: metadata[key] for key in CONFIG_KEYS if key in metadata}
    model_config = metadata.get("model_config") or {}
    for key in MODEL_CONFIG_KEYS:
        if key in model_config:
            config[f"model_{key}"] = model_config[key]
    return config


def init_smoke_run(
    *,
    entity: str | None = None,
    project: str | None = None,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    job_type: str = "smoke",
):
    require_api_key()
    import wandb

    run = wandb.init(
        entity=entity or os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
        project=project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        name=name or os.environ.get("WANDB_NAME"),
        job_type=job_type,
        config=dict(config or {}),
        settings={"console": "wrap"},
    )
    run.define_metric("stage/*", step_metric="stage/index")
    run.define_metric("eval/*", step_metric="eval/problem_index")
    run.define_metric("allocation/*", step_metric="allocation/layer")
    run.define_metric("sensitivity/*", step_metric="sensitivity/layer")
    return run


def log_stage(run, name: str, seconds: float, returncode: int, index: int) -> None:
    run.log(
        {
            "stage/index": index,
            "stage/name": name,
            "stage/seconds": seconds,
            "stage/returncode": returncode,
        }
    )
    run.summary[f"stage/{name}/seconds"] = seconds
    run.summary[f"stage/{name}/returncode"] = returncode


def save_live(path: Path, base: Path) -> None:
    if not path.is_file():
        return
    import wandb

    wandb.save(str(path), base_path=str(base), policy="live")


def _mean_bits(layer_counts: Mapping[str, Any]) -> float | None:
    total = sum(int(count) for count in layer_counts.values())
    if total <= 0:
        return None
    weighted = sum(int(bit) * int(count) for bit, count in layer_counts.items())
    return weighted / total


def log_allocation(run, budgets) -> None:
    import wandb

    rows = [[index, float(budget)] for index, budget in enumerate(budgets)]
    table = wandb.Table(data=rows, columns=["layer", "budget_mib"])
    run.log(
        {
            "allocation/per_layer": table,
            "allocation/budget_mib": wandb.plot.bar(
                table, "layer", "budget_mib", title="Per-layer KV budget (MiB)"
            ),
        }
    )
    for index, budget in enumerate(budgets):
        run.log({"allocation/layer": index, "allocation/budget_mib_point": float(budget)})
    run.summary["allocation/total_mib"] = float(sum(float(budget) for budget in budgets))
    run.summary["allocation/n_layers"] = len(rows)
    run.summary["allocation/mean_mib"] = (
        run.summary["allocation/total_mib"] / len(rows) if rows else 0.0
    )


def log_sensitivity(run, sensitivity: Mapping[str, Any]) -> None:
    import wandb

    for cache_name in ("k_sensitivity", "v_sensitivity"):
        by_bits = sensitivity.get(cache_name) or {}
        for bits, values in by_bits.items():
            rows = [[index, float(value)] for index, value in enumerate(values)]
            table = wandb.Table(data=rows, columns=["layer", "sensitivity"])
            title = f"{cache_name[0].upper()} sensitivity @ {bits}-bit"
            run.log(
                {
                    f"sensitivity/{cache_name}/{bits}bit": table,
                    f"sensitivity/{cache_name}/{bits}bit_plot": wandb.plot.line(
                        table, "layer", "sensitivity", title=title
                    ),
                }
            )


def log_summary(run, summary: Mapping[str, Any]) -> None:
    for method, stats in summary.items():
        correct = int(stats.get("correct", 0))
        responses = int(stats.get("responses", 0))
        accuracy = correct / responses if responses else 0.0
        run.summary[f"eval/{method}/correct"] = correct
        run.summary[f"eval/{method}/responses"] = responses
        run.summary[f"eval/{method}/accuracy"] = accuracy
        run.summary[f"eval/{method}/pass_at_1"] = 100.0 * accuracy
        run.summary[f"eval/{method}/token_limit_hits"] = int(stats.get("token_limit_hits", 0))
        run.summary[f"eval/{method}/quantization_observed"] = bool(
            stats.get("quantization_observed")
        )
        for key, tokens in (stats.get("output_tokens") or {}).items():
            run.summary[f"eval/{method}/output_tokens/{key}"] = int(tokens)


def log_eval_records(run, method: str, records: Mapping[str, Any]) -> None:
    mean_kv_bits = []
    for problem_index, (key, record) in enumerate(records.items()):
        metrics = {
            "eval/problem_index": problem_index,
            "eval/problem": key,
            f"eval/{method}/problem_correct": int(bool(record.get("judgement"))),
            f"eval/{method}/problem_output_len": int(record.get("output_len") or 0),
            f"eval/{method}/problem_input_len": int(record.get("input_len") or 0),
            f"eval/{method}/problem_hit_token_limit": int(bool(record.get("hit_token_limit"))),
        }
        if record.get("elapsed_seconds") is not None:
            metrics[f"eval/{method}/problem_elapsed_seconds"] = float(record["elapsed_seconds"])
        bit_counts = record.get("kv_bit_counts_by_layer") or []
        layer_bits = [
            bits for layer in bit_counts if (bits := _mean_bits(layer)) is not None
        ]
        if layer_bits:
            problem_bits = sum(layer_bits) / len(layer_bits)
            metrics[f"eval/{method}/problem_mean_kv_bits"] = problem_bits
            mean_kv_bits.append(problem_bits)
        run.log(metrics)
    if mean_kv_bits:
        run.summary[f"eval/{method}/mean_kv_bits"] = sum(mean_kv_bits) / len(mean_kv_bits)


def log_artifacts(run, output_dir: Path) -> None:
    import wandb

    artifact = wandb.Artifact(name=f"smoke-{run.id}", type="smoke-results")
    added = False
    for name in ARTIFACT_FILES:
        path = output_dir / name
        if path.is_file():
            artifact.add_file(str(path), name=name)
            added = True
    for method in ("original", "pm-kvq"):
        method_dir = output_dir / method
        if not method_dir.is_dir():
            continue
        for path in sorted(method_dir.glob("*.json")):
            artifact.add_file(str(path), name=f"{method}/{path.name}")
            added = True
    if added:
        run.log_artifact(artifact)


def _load_tensor_file(path: Path):
    if not path.is_file():
        return None
    import torch

    return torch.load(path, map_location="cpu")


def log_smoke_outputs(run, output_dir: Path, metadata: Mapping[str, Any] | None = None,
                      summary: Mapping[str, Any] | None = None) -> None:
    output_dir = Path(output_dir)
    if metadata is None and (output_dir / "metadata.json").is_file():
        metadata = json.loads((output_dir / "metadata.json").read_text())
    if summary is None and (output_dir / "summary.json").is_file():
        summary = json.loads((output_dir / "summary.json").read_text())
    if metadata:
        run.config.update(wandb_config_from_metadata(metadata), allow_val_change=True)
        run.summary["status"] = metadata.get("status")
        run.summary["output_dir"] = str(output_dir)
    budgets = _load_tensor_file(output_dir / "budgets.pt")
    if budgets is not None:
        log_allocation(run, budgets)
    sensitivity = _load_tensor_file(output_dir / "sensitivity.pt")
    if isinstance(sensitivity, dict):
        log_sensitivity(run, sensitivity)
    for method in ("original", "pm-kvq"):
        records = {}
        method_dir = output_dir / method
        if method_dir.is_dir():
            for path in sorted(method_dir.glob("*.json")):
                records.update(json.loads(path.read_text()))
        if records:
            log_eval_records(run, method, records)
    if summary:
        log_summary(run, summary)
    log_artifacts(run, output_dir)


def publish_existing_run(
    output_dir: Path,
    *,
    entity: str | None = None,
    project: str | None = None,
    name: str | None = None,
) -> str:
    output_dir = Path(output_dir).resolve()
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No metadata.json in {output_dir}")
    metadata = json.loads(metadata_path.read_text())
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    run = init_smoke_run(
        entity=entity,
        project=project,
        name=name or f"smoke-{output_dir.name}",
        config=wandb_config_from_metadata(metadata),
        job_type="smoke",
    )
    try:
        for index, stage in enumerate(metadata.get("stages") or [], start=1):
            log_stage(run, stage["name"], float(stage["seconds"]), int(stage["returncode"]), index)
            log_path = output_dir / f"{stage['name']}.log"
            save_live(log_path, output_dir)
        log_smoke_outputs(run, output_dir, metadata, summary)
        exit_code = 0 if metadata.get("status") == "complete" else 1
        run.finish(exit_code=exit_code)
        return run.url
    except BaseException:
        run.finish(exit_code=1)
        raise
