"""Request-local cache with absolute positions and physically compacted KV tensors."""

from collections import Counter

import torch
from transformers.cache_utils import DynamicCache

from .formats import quantize_dequantize

RETENTION = (64, 32, 16, 8, 4)
PRIORITY = {"transition": 0, "execution": 1, "reasoning": 2}


def attention_sparsity(weights, valid=None):
    """Head/row mean of entries strictly below 1% of the valid row maximum."""
    if valid is None:
        valid = torch.ones_like(weights, dtype=torch.bool)
    valid = valid.expand_as(weights)
    count = valid.sum(-1)
    if (count == 0).any():
        raise ValueError("attention sparsity has an empty valid row")
    maximum = weights.masked_fill(~valid, 0).amax(-1, keepdim=True)
    return (((weights < .01 * maximum) & valid).sum(-1).float() / count).mean().item()


def representative_indices(keys, keep, iterations=20):
    """Lloyd K-means, farthest-first seeds, then unique nearest original tokens.

    Flatten KV heads per token, use FP32 CPU distances, break ties by position.
    Empty clusters retain their seed; greedy unique selection handles duplicates.
    """
    n = keys.shape[-2]
    if not 0 < keep <= n:
        raise ValueError("representative count must be between one and segment size")
    points = keys.detach().float().cpu().squeeze(0).transpose(0, 1).reshape(n, -1)
    if not torch.isfinite(points).all():
        raise ValueError("nonfinite keys in K-means")
    if keep == n:
        return torch.arange(n, device=keys.device)
    selected = [0]
    distances = ((points - points[0]) ** 2).sum(-1)
    for _ in range(1, keep):
        distances[selected] = -1
        index = int(distances.argmax())
        selected.append(index)
        distances = torch.minimum(distances, ((points - points[index]) ** 2).sum(-1))
    centers = points[selected].clone()
    labels = None
    for _ in range(iterations):
        distances = ((points[:, None] - centers[None]) ** 2).sum(-1)
        next_labels = distances.argmin(-1)
        if labels is not None and torch.equal(labels, next_labels):
            break
        labels = next_labels
        for cluster in range(keep):
            members = points[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(0)
    selected = []
    for cluster in range(keep):
        distances = ((points - centers[cluster]) ** 2).sum(-1)
        # Prefer actual cluster members; empty clusters use any unused point.
        members = labels == cluster
        members[selected] = False
        if members.any():
            distances[~members] = torch.inf
        distances[selected] = torch.inf
        selected.append(int(distances.argmin()))
    return torch.tensor(sorted(selected), device=keys.device)


class ThinKVCache(DynamicCache):
    def __init__(self, config, num_layers, collect_traces=False):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.total_seen = 0
        self.prompt_length = 0
        self.positions = []
        self.precisions = []
        self.segments = []
        self.latest_sparsity = None
        self.step_sparsities = {}
        self.thought_counts = Counter()
        self.quantized_tokens = [Counter() for _ in range(num_layers)]
        self.evicted_tokens = [0] * num_layers
        self.eviction_events = [0] * num_layers
        self.collect_traces = collect_traces
        self.traces = {i: [] for i in config.selected_layers} if collect_traces else None
        self.in_step = False
        self.failed = False

    def get_seq_length(self, layer_idx=0):
        # HF generation uses this for input slicing/RoPE, never physical length.
        return self.total_seen

    def begin_step(self, length):
        if self.failed or self.in_step:
            raise ValueError("ThinKV cache is failed or already in a forward pass")
        if self.total_seen and length != 1:
            raise ValueError("ThinKV supports single-token decode only")
        self.is_prefill = self.total_seen == 0
        self.step_length = length
        self.step_sparsities = {}
        if self.is_prefill:
            self.prompt_length = length
            if self.config.eviction and length > self.config.token_budget:
                raise MemoryError("ThinKV token budget cannot accommodate protected prompt tokens")
        else:
            generated = self.total_seen - self.prompt_length
            if generated % self.config.refresh_interval == 0:
                if self.latest_sparsity is None:
                    raise ValueError("ThinKV is missing completed-forward sparsity statistics")
                self.segments.append({"start": self.total_seen, "end": self.total_seen,
                                      "thought": self.config.classify(self.latest_sparsity)})
            self.segments[-1]["end"] += 1
            self.thought_counts[self.segments[-1]["thought"]] += 1
        self.in_step = True

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if not self.in_step:
            raise ValueError("ThinKV cache update requires begin_step")
        position = torch.arange(self.total_seen, self.total_seen + self.step_length, device=key_states.device)
        precision = torch.full_like(position, 16)
        if layer_idx == len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.positions.append(position)
            self.precisions.append(precision)
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
            self.positions[layer_idx] = torch.cat((self.positions[layer_idx], position))
            self.precisions[layer_idx] = torch.cat((self.precisions[layer_idx], precision))
        if self.config.quantization and not self.is_prefill:
            segment = self.segments[-1]
            size = segment["end"] - segment["start"]
            group = self.config.group_size
            if size % group == 0:
                # Active tokens are protected, so the last group is contiguous.
                bits = self.config.bits(segment["thought"])
                self.key_cache[layer_idx][..., -group:, :] = quantize_dequantize(
                    self.key_cache[layer_idx][..., -group:, :], bits, group, axis=-2)
                self.value_cache[layer_idx][..., -group:, :] = quantize_dequantize(
                    self.value_cache[layer_idx][..., -group:, :], bits, group, axis=-1)
                self.precisions[layer_idx][-group:] = bits
                self.quantized_tokens[layer_idx][str(bits)] += group
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _segment_indices(self, layer, segment):
        positions = self.positions[layer]
        return torch.where((positions >= segment["start"]) & (positions < segment["end"]))[0]

    def _shrink(self, layer, segment):
        indices = self._segment_indices(layer, segment)
        target = next((n for n in RETENTION if n < len(indices)), None)
        if target is None:
            return False
        keys = self.key_cache[layer].index_select(-2, indices)
        representatives = indices[representative_indices(keys, target)]
        keep = torch.ones(len(self.positions[layer]), dtype=torch.bool, device=indices.device)
        keep[indices] = False
        keep[representatives] = True
        retained = torch.where(keep)[0]
        self.key_cache[layer] = self.key_cache[layer].index_select(-2, retained)
        self.value_cache[layer] = self.value_cache[layer].index_select(-2, retained)
        self.positions[layer] = self.positions[layer][retained]
        self.precisions[layer] = self.precisions[layer][retained]
        self.evicted_tokens[layer] += len(indices) - target
        self.eviction_events[layer] += 1
        return True

    def finish_step(self):
        if set(self.step_sparsities) != set(self.config.selected_layers):
            raise ValueError("ThinKV did not observe all selected layers in the completed forward")
        self.latest_sparsity = sum(self.step_sparsities.values()) / len(self.step_sparsities)
        if self.collect_traces and not self.is_prefill:
            for layer, scalar in self.step_sparsities.items():
                self.traces[layer].append(scalar)
        self.total_seen += self.step_length
        self.in_step = False
        if self.config.eviction and not self.is_prefill:
            active = self.segments[-1]
            preceding = self.segments[:-1]
            transition_completed = active["thought"] == "transition" and active["end"] - active["start"] == self.config.refresh_interval
            candidates = sorted(preceding, key=lambda s: (PRIORITY[s["thought"]], s["start"]))
            for layer in range(self.num_layers):
                if transition_completed:
                    for segment in preceding:
                        self._shrink(layer, segment)
                while len(self.positions[layer]) > self.config.token_budget:
                    if not any(self._shrink(layer, segment) for segment in candidates):
                        raise MemoryError(
                            "ThinKV token budget exhausted: protected prompt/active segment "
                            "and minimum segment retention cannot fit "
                            f"(budget={self.config.token_budget}, prompt={self.prompt_length}, "
                            f"active={active['end'] - active['start']}, segments={len(preceding)})")

    def diagnostics(self):
        return {
            "storage": "input_dtype_qdq_reference", "packed_quantized_storage": False,
            "total_seen_tokens": self.total_seen, "prompt_tokens": self.prompt_length,
            "decoded_tokens_cached": self.total_seen - self.prompt_length,
            "thought_counts": dict(self.thought_counts),
            "quantized_token_counts_by_layer": [dict(c) for c in self.quantized_tokens],
            "retained_precision_counts_by_layer": [
                {str(int(v)): int(c) for v, c in zip(*p.unique(return_counts=True))} for p in self.precisions],
            "evicted_tokens_by_layer": list(self.evicted_tokens),
            "eviction_events_by_layer": list(self.eviction_events),
            "retained_tokens_by_layer": [len(p) for p in self.positions],
            "cache_tensor_bytes": sum(t.numel() * t.element_size() for t in self.key_cache + self.value_cache),
            "metadata_tensor_bytes": sum(t.numel() * t.element_size() for t in self.positions + self.precisions),
            "cache_dtypes": sorted({str(t.dtype) for t in self.key_cache}),
        }

    def reorder_cache(self, beam_idx):
        raise ValueError("ThinKV does not support beam search or cache reordering")

    def batch_repeat_interleave(self, repeats):
        raise ValueError("ThinKV supports batch one only")

    def crop(self, max_length):
        raise ValueError("ThinKV does not support speculative decoding/cache cropping")

    def to_legacy_cache(self):
        raise ValueError("ThinKV requires its request-local cache; legacy caches are unsupported")
