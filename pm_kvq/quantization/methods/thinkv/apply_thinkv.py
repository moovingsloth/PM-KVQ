"""Small instance-local adapter for the installed Transformers 4.51.3 models."""

from copy import deepcopy
from dataclasses import replace
from functools import wraps
import inspect
from types import MethodType

import torch
import torch.nn.functional as F
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from pm_kvq.utils.modeling_utils import get_llm_layers, get_model_type
from .cache import ThinKVCache, attention_sparsity
from .config import load_calibration, model_identity, numerical_settings


def thinkv_attention(self, hidden_states, position_embeddings, attention_mask,
                     past_key_value=None, cache_position=None, **kwargs):
    if not isinstance(past_key_value, ThinKVCache):
        raise ValueError("ThinKV attention requires a ThinKVCache")
    cache = past_key_value
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
    key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    key, value = cache.update(key, value, self.layer_idx)
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    # The caller validates an unpadded, contiguous absolute-position sequence.
    # Build causality against retained ORIGINAL positions, not compact indices.
    valid = cache.positions[self.layer_idx][None, :] <= cache_position[:, None]
    if cache.is_prefill:
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=valid, scale=self.scaling)
        # Only the last prefill row seeds the first decode segment.
        statistic_query = query[..., -1:, :]
        statistic_valid = valid[-1:]
    else:
        statistic_query = query
        statistic_valid = valid
    weights = None
    if not cache.is_prefill or self.layer_idx in cache.config.selected_layers:
        scores = (statistic_query @ key.transpose(-2, -1)) * self.scaling
        scores = scores.masked_fill(~statistic_valid, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if self.layer_idx in cache.config.selected_layers:
        cache.step_sparsities[self.layer_idx] = attention_sparsity(weights, statistic_valid)
    if not cache.is_prefill:
        output = weights @ value
    output = output.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
    return self.o_proj(output), None


def apply_thinkv(model, thinkv_calibration=None, thinkv_token_budget=1024,
                thinkv_refresh_interval=None, thinkv_reasoning_bits=None,
                thinkv_execution_bits=None, thinkv_transition_bits=None,
                *, config=None, collect_traces=False):
    """Install inference adapter. ``config`` is for controlled tests/calibration.

    Production evaluation must supply a validated artifact; no fallback thresholds.
    No cache tensors are kept on the model between requests, only scalar diagnostics.
    """
    get_model_type(model)
    layers = get_llm_layers(model)
    if transformers.__version__ != "4.51.3":
        raise ValueError("ThinKV reference supports installed Transformers 4.51.3 only")
    if hasattr(model, "thinkv_config"):
        raise ValueError("ThinKV is already installed on this model")
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("ThinKV does not support sliding-window attention")
    if config is None:
        if thinkv_calibration is None:
            raise ValueError("--thinkv_calibration is required for ThinKV evaluation")
        artifact, config = load_calibration(
            thinkv_calibration, model_identity(model.config, model.config._name_or_path))
        overrides = {"token_budget": thinkv_token_budget}
        for name, value in (("refresh_interval", thinkv_refresh_interval),
                            ("reasoning_bits", thinkv_reasoning_bits),
                            ("execution_bits", thinkv_execution_bits),
                            ("transition_bits", thinkv_transition_bits)):
            if value is not None:
                overrides[name] = value
        config = replace(config, **overrides)
        model.thinkv_calibration_metadata = artifact
    if max(config.selected_layers) >= len(layers):
        raise ValueError("ThinKV selected layer exceeds model layer count")
    model.thinkv_config = config
    model.thinkv_last_diagnostics = None
    model.config._attn_implementation = "eager"
    original_forward = model.forward
    forward_signature = inspect.signature(original_forward)
    original_generate = model.generate
    generate_signature = inspect.signature(original_generate)

    @wraps(original_forward)
    def forward(*args, **kwargs):
        bound = forward_signature.bind(*args, **kwargs)
        values = bound.arguments
        model.thinkv_last_diagnostics = None
        if model.training or values.get("labels") is not None:
            raise ValueError("ThinKV reference supports inference with model.eval() only")
        if values.get("use_cache", model.config.use_cache) is False:
            raise ValueError("ThinKV requires use_cache=True")
        if values.get("output_attentions", model.config.output_attentions):
            raise ValueError("ThinKV collects scalar sparsity; output_attentions is unsupported")
        inputs = values.get("input_ids")
        if inputs is None or inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] == 0 or values.get("inputs_embeds") is not None:
            raise ValueError("ThinKV requires nonempty batch-one input_ids; inputs_embeds is unsupported")
        cache = values.get("past_key_values")
        if cache is None:
            cache = ThinKVCache(config, len(layers), collect_traces)
        if not isinstance(cache, ThinKVCache) or cache.config != config or cache.num_layers != len(layers):
            raise ValueError("ThinKV rejects external, legacy, static, and incompatible caches")
        expected = torch.arange(cache.total_seen, cache.total_seen + inputs.shape[1], device=inputs.device)
        mask = values.get("attention_mask")
        if mask is not None and (mask.ndim != 2 or tuple(mask.shape) != (1, cache.total_seen + inputs.shape[1]) or not (mask == 1).all()):
            raise ValueError("ThinKV requires an all-ones absolute-length mask; padding/custom masks are unsupported")
        for name, expected_value in (("cache_position", expected), ("position_ids", expected[None])):
            given = values.get(name)
            if given is not None and (given.shape != expected_value.shape or not torch.equal(given.to(inputs.device), expected_value)):
                raise ValueError(f"ThinKV requires contiguous absolute {name}")
            values[name] = expected_value
        values["past_key_values"] = cache
        values["use_cache"] = True
        try:
            cache.begin_step(inputs.shape[1])
            with torch.no_grad():
                output = original_forward(*bound.args, **bound.kwargs)
                cache.finish_step()
        except BaseException:
            cache.failed = True
            raise
        model.thinkv_last_diagnostics = cache.diagnostics()
        model.thinkv_last_diagnostics["settings"] = dict(numerical_settings(config), token_budget=config.token_budget)
        return output

    @wraps(original_generate)
    def generate(*args, **kwargs):
        bound = generate_signature.bind(*args, **kwargs)
        values = bound.arguments
        extra = values.setdefault("kwargs", {})
        generation_config = deepcopy(values.get("generation_config") or model.generation_config)
        generation_config.update(**extra)
        mode = generation_config.get_generation_mode(values.get("assistant_model"))
        if mode.value not in ("greedy_search", "sample") or generation_config.num_return_sequences != 1:
            raise ValueError("ThinKV supports batch-one greedy/sampling generation; beams/assisted modes are unsupported")
        if generation_config.cache_implementation is not None or not generation_config.use_cache or generation_config.return_legacy_cache:
            raise ValueError("ThinKV requires its request-local cache and use_cache=True")
        if generation_config.token_healing or generation_config.output_attentions:
            raise ValueError("ThinKV does not support token healing or attention-matrix outputs")
        if extra.get("past_key_values") is not None:
            raise ValueError("ThinKV generate starts a fresh request; external caches are unsupported")
        cache = ThinKVCache(config, len(layers), collect_traces)
        extra["past_key_values"] = cache
        model.thinkv_last_diagnostics = None
        try:
            output = original_generate(*bound.args, **bound.kwargs)
        except BaseException:
            model.thinkv_last_diagnostics = None
            raise
        sequences = output.sequences if hasattr(output, "sequences") else output
        model.thinkv_last_diagnostics["generated_sequence_length"] = sequences.shape[-1]
        model.thinkv_last_diagnostics["output_tokens"] = sequences.shape[-1] - cache.prompt_length
        if collect_traces:
            model.thinkv_last_traces = cache.traces
        return output

    for layer in layers:
        layer.self_attn.forward = MethodType(thinkv_attention, layer.self_attn)
    model.forward = forward
    model.generate = generate
