import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from pm_kvq.quantization.methods.quant_wrapper import quantize_model
from pm_kvq.quantization.methods.thinkv import ThinKVConfig, apply_thinkv
from pm_kvq.quantization.methods.thinkv.cache import ThinKVCache, attention_sparsity, representative_indices
from pm_kvq.quantization.methods.thinkv.calibration import KDE_SETTINGS, calibrate_traces
from pm_kvq.quantization.methods.thinkv.config import load_calibration, model_identity, numerical_settings, validate_calibration
from pm_kvq.quantization.methods.thinkv.formats import quantize_dequantize

torch.set_num_threads(1)


def artifact_for(model):
    return dict(schema_version=2, model=model_identity(model.config, model.config._name_or_path),
                thresholds=[.3, .7], selected_layers=[0, 1, 2, 3], seed=42,
                numerical=numerical_settings(ThinKVConfig()),
                dataset=dict(path='/local/s1k', fingerprint='test-fixture', prompt_field='question',
                             indices=[0], prompt_sha256='a' * 64),
                generation=dict(max_new_tokens=32, do_sample=True, temperature=.6, top_p=.95),
                kde=dict(KDE_SETTINGS), diagnostics={str(i): {'mode_counts': [3], 'errors': [None]} for i in range(4)},
                versions=dict(torch=torch.__version__, transformers='4.51.3', scipy='1.15.3', python='3.10.0'))


class FormatsTests(unittest.TestCase):
    def test_nvfp4_levels_and_midpoints(self):
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.] * 2)
        torch.testing.assert_close(quantize_dequantize(levels, 4), levels)
        x = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.] * 2)
        expected = torch.tensor([0., 1., 1., 2., 2., 4., 4., 6.] * 2)
        torch.testing.assert_close(quantize_dequantize(x, 4), expected)
        torch.testing.assert_close(quantize_dequantize(-x, 4), -expected)

    def test_fp4_either_side_of_every_midpoint(self):
        midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
        lower = torch.tensor([0., .5, 1., 1.5, 2., 3., 4.])
        upper = torch.tensor([.5, 1., 1.5, 2., 3., 4., 6.])
        for sign in (1, -1):
            for offset, expected in ((-.0001, lower), (.0001, upper)):
                # Keep group maximum at 6, making the NVFP4 scale exactly one.
                x = torch.cat([midpoints + offset, torch.tensor([6.])] * 2) * sign
                target = torch.cat([expected, torch.tensor([6.])] * 2) * sign
                torch.testing.assert_close(quantize_dequantize(x, 4), target)

    def test_ternary_zero_and_incomplete(self):
        x = torch.tensor([-1, -.6, -.4, 0, .4, .6, 1, 0.] * 2 + [.123])
        result = quantize_dequantize(x, 2)
        self.assertEqual(set(result[:16].tolist()), {-1., 0., 1.})
        self.assertEqual(result[-1], x[-1])
        for bits in (2, 4, 8):
            torch.testing.assert_close(quantize_dequantize(torch.zeros(2, 32), bits), torch.zeros(2, 32))

    def test_scale_representation_and_axes(self):
        x = torch.cat([torch.full((16,), 6.), torch.full((16,), 3.1)])
        scale = torch.tensor(3.1 / 6 * 448).to(torch.float8_e4m3fn).float() / 448
        self.assertAlmostEqual(quantize_dequantize(x, 4)[-1].item(), (scale * 6).item(), places=6)
        x = torch.arange(32 * 18).float().reshape(1, 1, 32, 18) / 17
        k = quantize_dequantize(x, 4, axis=-2)
        v = quantize_dequantize(x, 4, axis=-1)
        self.assertFalse(torch.equal(k, v))
        torch.testing.assert_close(v[..., -2:], x[..., -2:])
        self.assertTrue(torch.isfinite(quantize_dequantize(x * 1e-20, 4)).all())

    def test_fp8_and_reject_nonfinite(self):
        x = torch.tensor([0., 1.1, -2.3, 448.])
        torch.testing.assert_close(quantize_dequantize(x, 8), x.to(torch.float8_e4m3fn).float())
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            quantize_dequantize(torch.tensor([float('nan')]), 4)


class CalibrationTests(unittest.TestCase):
    def test_boundaries_and_masked_sparsity(self):
        config = ThinKVConfig()
        self.assertEqual([config.classify(s) for s in (0, .3, .7, 1)],
                         ['execution', 'reasoning', 'transition', 'transition'])
        scores = torch.tensor([[[[10., 0., 0., float('nan')]]]])
        valid = torch.tensor([[True, True, True, False]])
        self.assertAlmostEqual(attention_sparsity(scores, valid), 2 / 3)

    def test_gqa_pooling_before_softmax_and_mha_mean_before_threshold(self):
        scores = torch.tensor([[[[20., 0., 0.]], [[0., 10., 0.]]]])
        # Opposing query heads: pooling logits preserves the larger peak.
        self.assertAlmostEqual(attention_sparsity(scores, num_key_value_groups=2), 2 / 3)
        # MHA averages probabilities first, retaining both heads' peaks.
        self.assertAlmostEqual(attention_sparsity(scores), 1 / 3)
        # Head offsets do not affect MHA softmax but must affect GQA logit pooling.
        shifted = scores.clone()
        shifted[:, 1] += 20
        self.assertAlmostEqual(attention_sparsity(shifted, num_key_value_groups=2), 2 / 3)
        self.assertAlmostEqual(attention_sparsity(shifted), 1 / 3)

    def test_multiple_kv_groups_masks_and_empty_rows(self):
        scores = torch.tensor([[[[20., 0., 0., 0.]], [[0., 10., 0., 0.]],
                                [[0., 0., 10., 0.]], [[0., 0., 0., 10.]]]])
        self.assertAlmostEqual(attention_sparsity(scores, num_key_value_groups=2), .25)
        valid = torch.tensor([[True, False, True, True]])
        self.assertEqual(attention_sparsity(scores, valid, 2), 0.)
        for dtype in (torch.float32, torch.bfloat16):
            uniform = torch.zeros(2, 4, 2, 4, dtype=dtype)
            mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
            self.assertEqual(attention_sparsity(uniform, mask, 2), 0.)
        with self.assertRaisesRegex(ValueError, 'empty valid row'):
            attention_sparsity(scores, torch.zeros(1, 4, dtype=torch.bool), 2)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            attention_sparsity(scores, num_key_value_groups=3)
        with self.assertRaisesRegex(ValueError, 'finite'):
            attention_sparsity(scores * float('nan'))

    def test_kde_three_modes_selection_and_failure(self):
        rng = np.random.default_rng(42)
        samples = np.concatenate([rng.normal(center, .008, 160) for center in (.15, .5, .85)]).tolist()
        result = calibrate_traces({i: [samples, samples] for i in (5, 3, 1, 4, 2)})
        self.assertEqual(result['selected_layers'], [1, 2, 3, 4])
        self.assertTrue(.25 < result['thresholds'][0] < .4)
        self.assertTrue(.6 < result['thresholds'][1] < .75)
        with self.assertRaisesRegex(ValueError, 'diagnostics=.*constant sparsity trace'):
            calibrate_traces({i: [[.5] * 30] for i in range(4)})
        with self.assertRaisesRegex(ValueError, 'qualified='):
            calibrate_traces({i: [samples, [.5] * 30] for i in range(4)})
        with self.assertRaisesRegex(ValueError, 'four layers'):
            calibrate_traces({i: [samples] for i in range(3)})

    def test_metadata_roundtrip_and_rejections(self):
        model = ModelTests.make_model(Qwen2Config, Qwen2ForCausalLM)
        data = artifact_for(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            path.write_text(json.dumps(data))
            loaded, config = load_calibration(path, data['model'])
            self.assertEqual(loaded, data)
            self.assertEqual(loaded['schema_version'], 2)
            self.assertEqual(config.selected_layers, (0, 1, 2, 3))
            quantize_model(model, 'thinkv', {'thinkv_calibration': str(path)})
            self.assertEqual(model.thinkv_config.token_budget, 1024)
        for key, value in [('thresholds', [.7, .3]), ('selected_layers', [0, 0, 2, 3]),
                           ('schema_version', True), ('seed', -1), ('versions', {})]:
            invalid = copy.deepcopy(data)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_calibration(invalid)
        invalid = copy.deepcopy(data)
        invalid['numerical']['group_size'] = 32
        with self.assertRaises(ValueError):
            validate_calibration(invalid)
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_calibration(data, dict(data['model'], path='/different-model'))
        with self.assertRaisesRegex(ValueError, 'recalibrate.*schema 2'):
            validate_calibration(dict(data, schema_version=1))


class CacheTests(unittest.TestCase):
    def step(self, cache, length=1, sparsity=.1):
        cache.begin_step(length)
        for layer in range(cache.num_layers):
            # Values encode original position, allowing representative-pair checks.
            positions = torch.arange(cache.total_seen, cache.total_seen + length).float()
            keys = positions[None, None, :, None].expand(1, 2, length, 16).clone()
            cache.update(keys, keys + 1000, layer)
        cache.step_sparsities = {i: sparsity for i in cache.config.selected_layers}
        cache.finish_step()

    def test_deterministic_representatives_with_duplicates(self):
        keys = torch.tensor([0., 0., 1., 1., 10., 10., 11., 11.])[None, None, :, None]
        selected = representative_indices(keys, 2)
        self.assertEqual(selected.tolist(), [0, 4])
        self.assertTrue(torch.equal(selected, representative_indices(keys, 2)))
        self.assertEqual(representative_indices(torch.zeros_like(keys), 4).tolist(), [0, 1, 2, 3])

    def test_transition_anneals_only_preceding_and_preserves_pairs(self):
        cache = ThinKVCache(ThinKVConfig(selected_layers=(0,), quantization=False, token_budget=4096), 1)
        self.step(cache, 3)
        # First execution; then five consecutive transition segments.
        for _ in range(128):
            self.step(cache, sparsity=.9)
        for target in (64, 32, 16, 8, 4):
            for _ in range(128):
                self.step(cache, sparsity=.9)
            self.assertEqual(len(cache._segment_indices(0, cache.segments[0])), target)
            self.assertEqual(len(cache._segment_indices(0, cache.segments[-1])), 128)
        self.assertEqual(cache.positions[0][:3].tolist(), [0, 1, 2])
        torch.testing.assert_close(cache.value_cache[0][0, 0, :, 0], cache.positions[0].float() + 1000)
        self.assertGreater(cache.evicted_tokens[0], 0)

    def test_budget_priority_and_oldest(self):
        cache = ThinKVCache(ThinKVConfig(selected_layers=(0,), quantization=False, token_budget=10000), 1)
        self.step(cache, 2, .1)
        for scalar in (.1, .9, .5, .5):
            for _ in range(128):
                self.step(cache, sparsity=scalar)
        # Types E, E, T, R; T completion already annealed the first two E's.
        before = [len(cache._segment_indices(0, s)) for s in cache.segments]
        from dataclasses import replace
        cache.config = replace(cache.config, token_budget=len(cache.positions[0]))
        self.step(cache, sparsity=.5)
        after = [len(cache._segment_indices(0, s)) for s in cache.segments]
        self.assertEqual(after[:2], before[:2])
        self.assertLess(after[2], before[2])
        self.assertEqual(after[3], before[3])
        # Tie between execution segments selects the older one.
        cache.segments[2]['thought'] = 'reasoning'
        cache.config = replace(cache.config, token_budget=len(cache.positions[0]))
        self.step(cache)
        self.assertLess(len(cache._segment_indices(0, cache.segments[0])), before[0])
        self.assertEqual(len(cache._segment_indices(0, cache.segments[1])), before[1])

    def test_budget_exhaustion_and_incomplete_groups(self):
        cache = ThinKVCache(ThinKVConfig(selected_layers=(0,), token_budget=10), 1)
        with self.assertRaisesRegex(MemoryError, 'prompt'):
            self.step(cache, 11)
        cache = ThinKVCache(ThinKVConfig(selected_layers=(0,), token_budget=10), 1)
        self.step(cache, 3)
        for _ in range(7):
            self.step(cache)
        self.assertFalse(cache.quantized_tokens[0])
        with self.assertRaisesRegex(MemoryError, 'minimum segment retention'):
            self.step(cache)

    def test_transition_ternary_and_optional_reasoning_fp8_activate(self):
        for scalar, bits in ((.9, '2'), (.5, '8')):
            with self.subTest(bits=bits):
                cache = ThinKVCache(ThinKVConfig(selected_layers=(0,), reasoning_bits=8), 1)
                self.step(cache, 3, sparsity=scalar)
                prompt = cache.key_cache[0].clone()
                for _ in range(15):
                    self.step(cache, sparsity=scalar)
                self.assertFalse(cache.quantized_tokens[0])
                self.step(cache, sparsity=scalar)
                self.assertEqual(cache.quantized_tokens[0], {bits: 16})
                torch.testing.assert_close(cache.key_cache[0][..., :3, :], prompt)
                original = torch.arange(3, 19).float()
                self.assertFalse(torch.equal(cache.key_cache[0][0, 0, -16:, 0], original))


class ModelTests(unittest.TestCase):
    families = ((Qwen2Config, Qwen2ForCausalLM), (LlamaConfig, LlamaForCausalLM))

    @staticmethod
    def make_model(config_cls, model_cls):
        torch.manual_seed(42)
        config = config_cls(vocab_size=64, hidden_size=64, intermediate_size=80,
                            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=1024, bos_token_id=1, eos_token_id=None,
                            pad_token_id=0, attention_dropout=0.)
        config._attn_implementation = 'eager'
        return model_cls(config).eval()

    def test_disabled_compression_baseline_parity(self):
        for config_cls, model_cls in self.families:
            with self.subTest(model=model_cls.__name__), torch.no_grad():
                original = self.make_model(config_cls, model_cls)
                model = copy.deepcopy(original)
                apply_thinkv(model, config=ThinKVConfig(quantization=False, eviction=False))
                ids = torch.tensor([[1, 7, 13, 9]])
                a, b = original(ids, use_cache=True), model(ids)
                torch.testing.assert_close(a.logits, b.logits, atol=1e-6, rtol=1e-5)
                for token in (4, 6, 11):
                    a = original(torch.tensor([[token]]), past_key_values=a.past_key_values, use_cache=True)
                    b = model(torch.tensor([[token]]), past_key_values=b.past_key_values)
                    torch.testing.assert_close(a.logits, b.logits, atol=1e-6, rtol=1e-5)
                kwargs = dict(max_new_tokens=8, do_sample=False)
                self.assertTrue(torch.equal(original.generate(ids, **kwargs), model.generate(ids, **kwargs)))

    def test_generation_quantization_eviction_and_isolation(self):
        for config_cls, model_cls in self.families:
            with self.subTest(model=model_cls.__name__):
                model = self.make_model(config_cls, model_cls)
                config = ThinKVConfig(thresholds=(0., .000001), refresh_interval=16, token_budget=28)
                apply_thinkv(model, config=config)
                ids = torch.tensor([[1, 7, 13, 9]])
                first = model.generate(ids, max_new_tokens=40, do_sample=False, return_dict_in_generate=True)
                first_diag = copy.deepcopy(model.thinkv_last_diagnostics)
                cache = first.past_key_values
                self.assertEqual(first.sequences.shape[-1], 44)
                self.assertEqual(cache.get_seq_length(), 43)  # Last sampled token is not forwarded.
                self.assertTrue(all(n <= 28 for n in first_diag['retained_tokens_by_layer']))
                self.assertTrue(all(n > 0 for n in first_diag['evicted_tokens_by_layer']))
                self.assertTrue(all(sum(c.values()) > 0 for c in first_diag['quantized_token_counts_by_layer']))
                self.assertEqual(sum(first_diag['thought_counts'].values()), 39)
                self.assertLess(first_diag['cache_tensor_bytes'], 43 * 4 * 2 * 2 * 16 * 4)
                for layer, positions in enumerate(cache.positions):
                    self.assertEqual(positions[:4].tolist(), [0, 1, 2, 3])
                    self.assertEqual(int(positions[-1]), 42)
                    self.assertTrue((positions[1:] > positions[:-1]).all())
                second = model.generate(ids, max_new_tokens=40, do_sample=False, return_dict_in_generate=True)
                self.assertIsNot(first.past_key_values, second.past_key_values)
                self.assertTrue(torch.equal(first.sequences, second.sequences))
                self.assertEqual(first_diag, model.thinkv_last_diagnostics)
                print(f'{model_cls.__name__}: quantized={first_diag["quantized_token_counts_by_layer"]}, '
                      f'evicted={first_diag["evicted_tokens_by_layer"]}, retained={first_diag["retained_tokens_by_layer"]}, '
                      f'actual_KV_bytes={first_diag["cache_tensor_bytes"]}')

    def test_positions_match_uncompacted_sparse_oracle(self):
        # An independent original attention model with zero/masked holes must agree.
        for config_cls, model_cls in self.families:
            with self.subTest(model=model_cls.__name__), torch.no_grad():
                original = self.make_model(config_cls, model_cls)
                model = copy.deepcopy(original)
                apply_thinkv(model, config=ThinKVConfig(quantization=False, refresh_interval=8, token_budget=16))
                output = model(torch.tensor([[1, 7, 9]]))
                for _ in range(20):
                    output = model(torch.tensor([[5]]), past_key_values=output.past_key_values)
                cache = output.past_key_values
                seen = cache.total_seen
                # Masks may differ by layer, so compare a single attention module.
                for layer in (0, 3):
                    attention = original.model.layers[layer].self_attn
                    old_positions = cache.positions[layer].clone()
                    k, v = cache.key_cache[layer].clone(), cache.value_cache[layer].clone()
                    holes = DynamicCache()
                    for _ in range(layer + 1):
                        holes.key_cache.append(torch.zeros(1, 2, seen, 16))
                        holes.value_cache.append(torch.zeros(1, 2, seen, 16))
                    holes.key_cache[layer][:, :, old_positions] = k
                    holes.value_cache[layer][:, :, old_positions] = v
                    position = torch.tensor([seen])
                    hidden = torch.randn(1, 1, 64)
                    embedding = original.model.rotary_emb(hidden, position[None])
                    mask = torch.full((1, 1, 1, seen + 1), torch.finfo(hidden.dtype).min)
                    mask[..., old_positions] = 0
                    mask[..., seen] = 0
                    expected = attention(hidden, embedding, mask, past_key_value=holes, cache_position=position)[0]
                    isolated = ThinKVCache(model.thinkv_config, 4)
                    isolated.total_seen, isolated.prompt_length = seen, cache.prompt_length
                    isolated.key_cache = [t.clone() for t in cache.key_cache]
                    isolated.value_cache = [t.clone() for t in cache.value_cache]
                    isolated.positions = [t.clone() for t in cache.positions]
                    isolated.precisions = [t.clone() for t in cache.precisions]
                    isolated.segments = copy.deepcopy(cache.segments)
                    isolated.latest_sparsity = cache.latest_sparsity
                    isolated.begin_step(1)
                    actual = model.model.layers[layer].self_attn(hidden, embedding, None,
                                                                past_key_value=isolated, cache_position=position)[0]
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_cpu_bf16_default_refresh_270_token_generation(self):
        for config_cls, model_cls in self.families:
            with self.subTest(model=model_cls.__name__):
                model = self.make_model(config_cls, model_cls).to(torch.bfloat16)
                apply_thinkv(model, config=ThinKVConfig(token_budget=160))
                ids = torch.tensor([[1, 7, 13, 9]])
                output = model.generate(ids, max_new_tokens=270, do_sample=False, return_dict_in_generate=True)
                diagnostics = model.thinkv_last_diagnostics
                self.assertEqual(output.sequences.shape[-1], 274)
                self.assertEqual(diagnostics['settings']['refresh_interval'], 128)
                self.assertEqual(diagnostics['decoded_tokens_cached'], 269)
                self.assertEqual(diagnostics['cache_dtypes'], ['torch.bfloat16'])
                self.assertTrue(all(n > 0 for n in diagnostics['evicted_tokens_by_layer']))
                self.assertTrue(all(sum(counts.values()) >= 256 for counts in diagnostics['quantized_token_counts_by_layer']))
                self.assertTrue(all(n <= 160 for n in diagnostics['retained_tokens_by_layer']))
                for positions, keys in zip(output.past_key_values.positions, output.past_key_values.key_cache):
                    self.assertEqual(positions[:4].tolist(), [0, 1, 2, 3])
                    self.assertEqual(positions[-1].item(), 272)
                    self.assertTrue((positions[1:] > positions[:-1]).all())
                    self.assertTrue(torch.isfinite(keys).all())

    def test_calibration_and_inference_share_raw_score_statistics(self):
        original = self.make_model(Qwen2Config, Qwen2ForCausalLM)
        calibration, inference = copy.deepcopy(original), copy.deepcopy(original)
        config = ThinKVConfig(quantization=False, eviction=False)
        apply_thinkv(calibration, config=config, collect_traces=True)
        apply_thinkv(inference, config=config)
        ids = torch.tensor([[1, 3, 9]])
        adapter = importlib.import_module('pm_kvq.quantization.methods.thinkv.apply_thinkv')
        with patch.object(adapter, 'attention_sparsity', wraps=attention_sparsity) as statistic:
            calibration.generate(ids, max_new_tokens=6, do_sample=False)
            calibration_scores = [call.args[0].clone() for call in statistic.call_args_list]
            self.assertEqual(statistic.call_count, 4 * 6)
            self.assertTrue(all(call.args[2] == 2 for call in statistic.call_args_list))
            statistic.reset_mock()
            inference.generate(ids, max_new_tokens=6, do_sample=False)
            self.assertEqual(statistic.call_count, len(calibration_scores))
            for expected, call in zip(calibration_scores, statistic.call_args_list):
                torch.testing.assert_close(call.args[0], expected)
        for layer in range(4):
            expected = [attention_sparsity(calibration_scores[step * 4 + layer], num_key_value_groups=2)
                        for step in range(1, 6)]
            self.assertEqual(calibration.thinkv_last_traces[layer], expected)
        self.assertEqual(calibration.thinkv_last_diagnostics, inference.thinkv_last_diagnostics)

    def test_diagnostics_and_traces_cleared_before_all_failed_entries(self):
        model = self.make_model(Qwen2Config, Qwen2ForCausalLM)
        apply_thinkv(model, config=ThinKVConfig(token_budget=8), collect_traces=True)
        ids = torch.tensor([[1, 2]])
        failures = [
            (ValueError, lambda: model.generate(ids, num_beams=2)),
            (TypeError, lambda: model.generate(ids, inputs=ids)),
            (ValueError, lambda: model.generate(ids.expand(2, -1), max_new_tokens=2)),
            (MemoryError, lambda: model.generate(ids, max_new_tokens=12, do_sample=False)),
            (MemoryError, lambda: model.generate(ids.repeat(1, 5), max_new_tokens=2)),
            (ValueError, lambda: model(ids, use_cache=False)),
            (TypeError, lambda: model(ids, input_ids=ids)),
            (ValueError, lambda: model(ids[:, :0])),
            (MemoryError, lambda: model(ids.repeat(1, 5))),
        ]
        for error, fail in failures:
            with self.subTest(error=error, fail=fail):
                model.generate(ids, max_new_tokens=3, do_sample=False)
                self.assertIsNotNone(model.thinkv_last_diagnostics)
                self.assertTrue(model.thinkv_last_traces[0])
                with self.assertRaises(error):
                    fail()
                self.assertIsNone(model.thinkv_last_diagnostics)
                self.assertIsNone(model.thinkv_last_traces)
        # A direct decode failure must also clear observations of earlier forwards.
        output = model(ids)
        for _ in range(6):
            output = model(torch.tensor([[3]]), past_key_values=output.past_key_values)
        self.assertTrue(model.thinkv_last_traces[0])
        with self.assertRaises(MemoryError):
            model(torch.tensor([[3]]), past_key_values=output.past_key_values)
        self.assertIsNone(model.thinkv_last_diagnostics)
        self.assertIsNone(model.thinkv_last_traces)

    def test_reject_unsupported_and_use_previous_completed_forward(self):
        model = self.make_model(Qwen2Config, Qwen2ForCausalLM)
        apply_thinkv(model, config=ThinKVConfig(refresh_interval=2, eviction=False))
        ids = torch.tensor([[1, 2]])
        for kwargs in ({'num_beams': 2}, {'num_return_sequences': 2, 'do_sample': True},
                       {'use_cache': False}, {'cache_implementation': 'static'}, {'output_attentions': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.generate(ids, max_new_tokens=2, **kwargs)
        for kwargs in ({'attention_mask': torch.tensor([[1, 0]])}, {'past_key_values': DynamicCache()},
                       {'position_ids': torch.tensor([[3, 4]])}, {'use_cache': False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(ids, **kwargs)
        with self.assertRaises(ValueError):
            model(ids.expand(2, -1))
        output = model(ids)
        cache = output.past_key_values
        cache.latest_sparsity = .9
        model(torch.tensor([[3]]), past_key_values=cache)
        self.assertEqual(cache.segments[-1]['thought'], 'transition')
        cache.latest_sparsity = 0
        model(torch.tensor([[4]]), past_key_values=cache)
        self.assertEqual(cache.segments[-1]['thought'], 'transition')
        cache.latest_sparsity = 0
        model(torch.tensor([[5]]), past_key_values=cache)
        self.assertEqual(cache.segments[-1]['thought'], 'execution')


if __name__ == '__main__':
    unittest.main()
