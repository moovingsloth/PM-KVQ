"""Offline calibration from scalar decode sparsity traces, one list per layer/prompt."""

import json

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

KDE_SETTINGS = {"bandwidth": "scott", "grid_size": 4096,
                "layer_rule": "three_modes_every_prompt_first_four"}


def calibrate_traces(traces):
    if not isinstance(traces, dict) or not traces:
        raise ValueError("calibration requires layer-indexed scalar traces")
    diagnostics, eligible, valleys = {}, [], {}
    prompt_count = len(next(iter(traces.values())))
    if prompt_count == 0:
        raise ValueError("calibration has no prompts")
    grid = np.linspace(0., 1., 4096)
    for layer in sorted(traces):
        prompts = traces[layer]
        if len(prompts) != prompt_count:
            raise ValueError("all calibration layers must contain the same prompts")
        counts, errors, thresholds = [], [], []
        for prompt in prompts:
            samples = np.asarray(prompt, dtype=np.float64)
            try:
                if samples.ndim != 1 or len(samples) < 3 or not np.isfinite(samples).all() or np.any((samples < 0) | (samples > 1)):
                    raise ValueError("need at least three finite scalar sparsities in [0, 1]")
                if np.ptp(samples) == 0:
                    raise ValueError("constant sparsity trace")
                density = gaussian_kde(samples, bw_method="scott")(grid)
                # Endpoints can be modes on the bounded sparsity domain.
                peaks = find_peaks(np.r_[0., density, 0.])[0] - 1
                counts.append(len(peaks))
                errors.append(None)
                if len(peaks) == 3:
                    thresholds.append([float(grid[a + np.argmin(density[a:b + 1])])
                                       for a, b in zip(peaks[:-1], peaks[1:])])
            except (ValueError, np.linalg.LinAlgError) as error:
                counts.append(0)
                errors.append(str(error))
        diagnostics[str(layer)] = {"mode_counts": counts, "errors": errors}
        if all(count == 3 for count in counts):
            eligible.append(layer)
            valleys[layer] = thresholds
    if len(eligible) < 4:
        raise ValueError("ThinKV calibration needs four layers with three modes on EVERY prompt; "
                         f"qualified={eligible}; diagnostics={json.dumps(diagnostics)}")
    selected = eligible[:4]
    thresholds = np.asarray([valleys[layer] for layer in selected]).mean(axis=(0, 1)).tolist()
    return {"thresholds": thresholds, "selected_layers": selected,
            "diagnostics": diagnostics, "kde": dict(KDE_SETTINGS)}
