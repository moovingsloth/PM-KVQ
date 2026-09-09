"""Weights & Biases helpers for PM-KVQ smoke and evaluation runs."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


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
    "preset",
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


def wandb_settings():
    """Force online mode. Console text is pushed with run.write_logs()."""
    import wandb

    os.environ["WANDB_MODE"] = "online"
    os.environ.pop("WANDB_DISABLED", None)
    os.environ.pop("WANDB_OFFLINE", None)
    return wandb.Settings(
        mode="online",
        console="off",
        symlink=False,
    )


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
        settings=wandb_settings(),
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


_ANSI_RE = re.compile(r"\x1b[@-Z\\-_]|\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS_RE = re.compile(r"%\||\d+\.\d+tok/s|\d+\.\d+s/(?:it|sample|problem)")


def emit_console(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def make_console_emitter(run=None):
    """Write to the terminal and to the W&B Logs tab while the run is online."""

    def emit(text: str) -> None:
        emit_console(text)
        if run is not None:
            run.write_logs(text)

    return emit


def collapse_cr(text: str) -> str:
    """Keep the last carriage-return state of each line so tqdm is readable online."""
    return "\n".join(part.rsplit("\r", 1)[-1] for part in text.split("\n"))


class ConsoleCollapser:
    """Fold tqdm/ANSI progress into the latest line so the W&B Logs tab stays readable."""

    def __init__(self, emit: Callable[[str], None] = emit_console):
        self.emit = emit
        self.buf = ""
        self.pending: str | None = None
        self.emitted_pending: str | None = None

    def feed(self, text: str, *, live: bool = False) -> None:
        self.buf += text.replace("\x1b[A", "\n")
        self._drain(final=False)
        if live and self.pending is not None and self.pending != self.emitted_pending:
            self.emit(self.pending + "\n")
            self.emitted_pending = self.pending

    def flush(self) -> None:
        self._drain(final=True)
        if self.pending is not None and self.pending != self.emitted_pending:
            self.emit(self.pending + "\n")
        self.pending = None
        self.emitted_pending = None

    def _drain(self, *, final: bool) -> None:
        while True:
            newline = self.buf.find("\n")
            carriage = self.buf.find("\r")
            if newline < 0 and carriage < 0:
                break
            if newline < 0:
                index = carriage
            elif carriage < 0:
                index = newline
            else:
                index = min(newline, carriage)
            line = _ANSI_RE.sub("", self.buf[:index])
            sep = self.buf[index]
            self.buf = self.buf[index + 1 :]
            if sep == "\r" and self.buf[:1] == "\n":
                self.buf = self.buf[1:]
            if sep == "\r" and not line.strip():
                continue
            self._accept(line.rstrip(" "))
        if final and self.buf:
            self._accept(_ANSI_RE.sub("", self.buf).rstrip(" "))
            self.buf = ""

    @staticmethod
    def _progress_key(line: str) -> str:
        return line.split(":", 1)[0].strip()

    def _accept(self, line: str) -> None:
        if not line.strip():
            return
        if _PROGRESS_RE.search(line):
            key = self._progress_key(line)
            if self.pending is not None and self._progress_key(self.pending) != key:
                if self.pending != self.emitted_pending:
                    self.emit(self.pending + "\n")
                self.emitted_pending = None
            self.pending = line
            return
        if self.pending is not None:
            if self.pending != self.emitted_pending:
                self.emit(self.pending + "\n")
            self.pending = None
            self.emitted_pending = None
        self.emit(line + "\n")


def collapse_progress(text: str) -> str:
    chunks: list[str] = []
    collapser = ConsoleCollapser(chunks.append)
    collapser.feed(text)
    collapser.flush()
    return "".join(chunks)


def replay_log_to_console(path: Path, emit: Callable[[str], None] = emit_console) -> None:
    if not path.is_file():
        return
    emit(f"\n===== {path.name} =====\n")
    text = path.read_bytes().decode("utf-8", "replace")
    emit(collapse_progress(text))


def copy_into_run(run, path: Path, relative: str | None = None) -> None:
    """Copy a file into the run directory and upload it now (no outside-dir symlink)."""
    if run is None or not path.is_file():
        return
    import wandb

    dest = Path(run.dir) / (relative or path.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != path.resolve():
        shutil.copy2(path, dest)
    wandb.save(str(dest), base_path=str(run.dir), policy="now", glob=False)


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


def _sens_list(by_bits: Mapping[Any, Any], bits: int) -> list[float] | None:
    if bits in by_bits:
        values = by_bits[bits]
    elif str(bits) in by_bits:
        values = by_bits[str(bits)]
    else:
        return None
    return [float(value) for value in values]


def _kv_sensitivity(sensitivity: Mapping[str, Any], bits: int) -> list[float] | None:
    keys = _sens_list(sensitivity.get("k_sensitivity") or {}, bits)
    values = _sens_list(sensitivity.get("v_sensitivity") or {}, bits)
    if keys is None or values is None or len(keys) != len(values):
        return None
    return [key + value for key, value in zip(keys, values)]


def _allocation_hparams(metadata: Mapping[str, Any] | None, budgets) -> tuple[int, int, float]:
    metadata = metadata or {}
    model = metadata.get("model_config") or {}
    heads = model.get("num_attention_heads")
    kv_heads = model.get("num_key_value_heads")
    hidden = model.get("hidden_size")
    kv_dim = kv_heads * hidden // heads if heads and kv_heads and hidden else 1024
    max_len = int(metadata.get("max_len") or 32768)
    if metadata.get("memory_budget_mb") is not None:
        memory_budget = float(metadata["memory_budget_mb"])
    elif budgets is not None:
        memory_budget = float(sum(float(budget) for budget in budgets))
    else:
        memory_budget = 0.0
    return int(kv_dim), max_len, memory_budget


def _fbit_from_mib(budget_mib: float, hidden_size: int, max_len: int) -> float:
    return float(budget_mib) * 8 * 1024 * 1024 / (hidden_size * 2 * max_len)


def _allocations_for_budgets(
    sensitivity: Mapping[str, Any],
    fbit_choices: list[int],
    hidden_size: int,
    max_len: int,
    memory_budgets: list[float],
) -> dict[float, dict[str, list]]:
    from pm_kvq.quantization.methods.pm_kvq.allocation.allocation import allocate_memory_budget

    k_sensitivity = {
        int(bits): [float(value) for value in values]
        for bits, values in (sensitivity.get("k_sensitivity") or {}).items()
    }
    v_sensitivity = {
        int(bits): [float(value) for value in values]
        for bits, values in (sensitivity.get("v_sensitivity") or {}).items()
    }
    results: dict[float, dict[str, list]] = {}
    for memory_budget in memory_budgets:
        try:
            fbits, layer_mib = allocate_memory_budget(
                fbit_choices, k_sensitivity, v_sensitivity, float(memory_budget),
                hidden_size, max_len,
            )
        except (ValueError, Exception):
            continue
        results[float(memory_budget)] = {
            "fbit": [int(bit) for bit in fbits],
            "mib": [float(value) for value in layer_mib],
        }
    return results


def _paper_budget_grid(n_layers: int, hidden_size: int, max_len: int, memory_budget: float,
                       fbit_choices: list[int]) -> list[float]:
    mib_by_bit = {bit: bit * hidden_size * 2 * max_len / 8 / 1024 / 1024 for bit in fbit_choices}
    lo = n_layers * min(mib_by_bit.values())
    hi = n_layers * max(mib_by_bit.values())
    grid = {round(lo, 6), round(hi, 6)}
    if memory_budget:
        grid.add(round(float(memory_budget), 6))
        if lo < memory_budget < hi:
            grid.add(round((lo + memory_budget) / 2, 6))
            grid.add(round((memory_budget + hi) / 2, 6))
    return sorted(budget for budget in grid if lo - 1e-6 <= budget <= hi + 1e-6)


def log_paper_figures(
    run,
    sensitivity: Mapping[str, Any],
    budgets=None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Log paper Fig. 3/4: per-block KV sensitivity and memory allocation by budget."""
    import wandb

    hidden_size, max_len, memory_budget = _allocation_hparams(metadata, budgets)
    fbit_choices = [4, 2]
    if budgets is not None:
        recovered = sorted({round(_fbit_from_mib(float(budget), hidden_size, max_len)) for budget in budgets}, reverse=True)
        if recovered:
            fbit_choices = [int(bit) for bit in recovered if bit > 0]

    n_layers = None
    sensitivity_by_bit: dict[int, list[float]] = {}
    for bits in sorted(set(fbit_choices) | {2, 4}):
        series = _kv_sensitivity(sensitivity, bits)
        if series:
            sensitivity_by_bit[bits] = series
            n_layers = len(series)
    if n_layers is None:
        return
    layers = list(range(n_layers))

    sensitivity_table = wandb.Table(columns=["layer"] + [f"s_kv_{bits}bit" for bits in sorted(sensitivity_by_bit)])
    for index in layers:
        sensitivity_table.add_data(index, *[sensitivity_by_bit[bits][index] for bits in sorted(sensitivity_by_bit)])
    run.log(
        {
            "paper/fig3_fig4_sensitivity_table": sensitivity_table,
            "paper/fig3_fig4_sensitivity": wandb.plot.line_series(
                xs=layers,
                ys=[sensitivity_by_bit[bits] for bits in sorted(sensitivity_by_bit)],
                keys=[f"{bits}-bit $s_{{i,b}}$" for bits in sorted(sensitivity_by_bit)],
                title="Fig. 3/4: KV-cache quantization sensitivity by transformer block",
                xname="transformer block",
            ),
        }
    )

    budget_grid = _paper_budget_grid(n_layers, hidden_size, max_len, memory_budget, fbit_choices)
    allocations = _allocations_for_budgets(
        sensitivity, fbit_choices, hidden_size, max_len, budget_grid,
    )
    if budgets is not None and memory_budget:
        allocations[float(memory_budget)] = {
            "fbit": [int(round(_fbit_from_mib(float(budget), hidden_size, max_len))) for budget in budgets],
            "mib": [float(budget) for budget in budgets],
        }
    if allocations:
        alloc_keys = [f"{budget:.0f} MiB" for budget in sorted(allocations)]
        run.log(
            {
                "paper/fig3_fig4_allocation": wandb.plot.line_series(
                    xs=layers,
                    ys=[allocations[budget]["fbit"] for budget in sorted(allocations)],
                    keys=alloc_keys,
                    title="Fig. 3/4: Block-wise Fbit allocation (color = total KV budget)",
                    xname="transformer block",
                ),
                "paper/fig3_fig4_memory": wandb.plot.line_series(
                    xs=layers,
                    ys=[allocations[budget]["mib"] for budget in sorted(allocations)],
                    keys=alloc_keys,
                    title="Fig. 3/4: Block-wise KV memory (color = total KV budget)",
                    xname="transformer block",
                ),
            }
        )
        alloc_table = wandb.Table(
            columns=["layer"] + [f"fbit_{budget:.0f}mib" for budget in sorted(allocations)]
            + [f"mib_{budget:.0f}mib" for budget in sorted(allocations)]
        )
        for index in layers:
            row = [index]
            for budget in sorted(allocations):
                row.append(allocations[budget]["fbit"][index])
            for budget in sorted(allocations):
                row.append(allocations[budget]["mib"][index])
            alloc_table.add_data(*row)
        run.log({"paper/fig3_fig4_allocation_table": alloc_table})

    image_path = _render_paper_figure(layers, sensitivity_by_bit, allocations, memory_budget)
    if image_path is not None:
        run.log({"paper/fig3_fig4": wandb.Image(
            str(image_path),
            caption="Paper Fig. 3/4: KV-cache quantization sensitivity by transformer block "
                    "and block-wise memory allocation. Different colors are different total KV budgets.",
        )})
        image_path.unlink(missing_ok=True)


def _render_paper_figure(layers, sensitivity_by_bit, allocations, memory_budget):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True, constrained_layout=True)
    ax_sens, ax_alloc = axes
    for bits, series in sorted(sensitivity_by_bit.items()):
        ax_sens.plot(layers, series, label=f"{bits}-bit", linewidth=1.6)
    ax_sens.set_ylabel(r"Sensitivity $s_{i,b}$")
    ax_sens.set_title("KV-cache quantization sensitivity")
    ax_sens.legend(frameon=False, fontsize=8)
    ax_sens.grid(True, alpha=0.3)

    if allocations:
        for budget, alloc in sorted(allocations.items()):
            highlight = memory_budget and abs(budget - float(memory_budget)) < 1e-3
            ax_alloc.step(
                layers,
                alloc["fbit"],
                where="mid",
                label=f"{budget:.0f} MiB",
                linewidth=2.2 if highlight else 1.4,
                alpha=1.0 if highlight else 0.85,
            )
        ax_alloc.set_ylabel("Allocated Fbit")
        ax_alloc.legend(frameon=False, fontsize=8, title="KV budget")
    ax_alloc.set_xlabel("Transformer block")
    ax_alloc.set_title("Block-wise memory allocation (color = total KV budget)")
    ax_alloc.grid(True, alpha=0.3)
    fig.suptitle("Paper Fig. 3 / Fig. 4", fontsize=12)

    path = Path("/tmp") / f"pmkvq_fig3_fig4_{os.getpid()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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
    for path in sorted(output_dir.glob("*.log")):
        artifact.add_file(str(path), name=f"logs/{path.name}")
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
        log_paper_figures(run, sensitivity, budgets, metadata=metadata or {})
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
        emit = make_console_emitter(run)
        emit(f"Publishing {output_dir} to {run.url}\n")
        for index, stage in enumerate(metadata.get("stages") or [], start=1):
            log_stage(run, stage["name"], float(stage["seconds"]), int(stage["returncode"]), index)
            log_path = output_dir / f"{stage['name']}.log"
            replay_log_to_console(log_path, emit=emit)
            copy_into_run(run, log_path, relative=f"stage_logs/{log_path.name}")
        for name in ("metadata.json", "summary.json"):
            copy_into_run(run, output_dir / name)
        log_smoke_outputs(run, output_dir, metadata, summary)
        exit_code = 0 if metadata.get("status") == "complete" else 1
        run.finish(exit_code=exit_code)
        return run.url
    except BaseException:
        run.finish(exit_code=1)
        raise
