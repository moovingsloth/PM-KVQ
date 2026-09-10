from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from pm_kvq.evaluation.aime_manifest import load_manifest, load_problems, manifest_bytes
from scripts.smoke_aime2025 import (
    ROOT, PRESETS, _aime_selection, _eval_jobs, _evaluation_options, _expected_keys,
    _judge_options, _method_options, build_parser, main,
)


class RunnerTests(unittest.TestCase):
    def test_preserved_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.methods, ['original', 'pm-kvq'])
        self.assertEqual(args.preset, 'smoke')
        self.assertEqual(args.aime_protocol, 'aime2025')
        self.assertIsNone(args.aime_dataset_path)
        self.assertTrue(args.wandb)
        self.assertEqual(PRESETS, {
            'smoke': {'n_samples': 8, 'n_responses': 1, 'slices': [(0, 1), (15, 16)]},
            'day': {'n_samples': 8, 'n_responses': 4, 'slices': [(0, 5), (15, 20)]},
            'full': {'n_samples': 512, 'n_responses': 16, 'slices': [(0, 30)]}})
        for name, count in [('smoke', 2), ('day', 40), ('full', 480)]:
            self.assertEqual(len(_expected_keys(PRESETS[name]['slices'], PRESETS[name]['n_responses'])), count)
        options = _method_options('pm-kvq', args, Path('/run'))
        self.assertEqual(options, ['--backend', 'fake', '--rep_scales', Path('/run/scales.pt'),
                                  '--kv_budgets', Path('/run/budgets.pt'), '--n_sink_token', 1,
                                  '--n_sink_token_bits', 16, '--n_window_token', 128,
                                  '--n_window_token_bits', 16, '--n_init_kv_bits', 16])
        self.assertEqual(_method_options('original', args, Path('/run')), [])

    def test_protocol_paths_and_command_construction(self):
        out = Path('/run')
        for protocol in ('aime2025', 'thinkv_mixed'):
            for preset in PRESETS:
                args = build_parser().parse_args(['--aime_protocol', protocol, '--preset', preset,
                                                  '--dataset_root', '/legacy'])
                selection, data = _aime_selection(args)
                self.assertEqual(data, Path('/legacy/aime') if protocol == 'aime2025' else ROOT / 'datasets/aime')
                selected = selection if protocol == 'thinkv_mixed' else None
                for method in ('original', 'pm-kvq', 'thinkv'):
                    jobs = _eval_jobs(method, PRESETS[preset]['slices'], out, selected)
                    self.assertEqual(len(jobs), 1 if selected else len(PRESETS[preset]['slices']))
                    judge = _judge_options(method, args, out)
                    for _, start, end, path in jobs:
                        options = _evaluation_options(method, args, out, '/model', data, start, end, path)
                        self.assertEqual(options[options.index('--n_responses') + 1], PRESETS[preset]['n_responses'])
                        self.assertEqual(options[options.index('--seed') + 1], 42)
                        self.assertEqual(options[options.index('--dataset_path') + 1], data)
                        if selected:
                            manifest = options[options.index('--aime_manifest') + 1]
                            self.assertEqual(manifest, out / 'aime_manifest.json')
                            self.assertEqual(judge[judge.index('--aime_manifest') + 1], manifest)
                            self.assertEqual(judge[judge.index('--n_responses') + 1], PRESETS[preset]['n_responses'])
                            self.assertNotIn('--version', options)
                            self.assertNotIn('--start', options)
                        else:
                            self.assertEqual(options[options.index('--version') + 1], 2025)
                            self.assertEqual(options[options.index('--start') + 1], start)
                            self.assertEqual(options[options.index('--end') + 1], end)
                            self.assertNotIn('--aime_manifest', options)
                args.aime_dataset_path = '/explicit/aime'
                self.assertEqual(_aime_selection(args)[1], Path('/explicit/aime'))

    def test_missing_data_fails_before_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['smoke_aime2025.py', '--methods', 'original', '--aime_protocol', 'thinkv_mixed',
                    '--aime_dataset_path', directory, '--no-wandb']
            with patch.object(sys, 'argv', argv), patch('torch.cuda.is_available') as cuda:
                with self.assertRaises(FileNotFoundError):
                    main()
                cuda.assert_not_called()

    def test_mixed_runner_writes_exact_coverage_and_provenance(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, model, out = root / 'data', root / 'model', root / 'run'
            for year in (2024, 2025):
                subset = data / f'aime_{year}_I'
                (subset / 'problems').mkdir(parents=True)
                (subset / 'problems/1.tex').write_text(f'Synthetic {year} problem')
                (subset / 'answers.csv').write_text('id,answer\n1,042\n')
            model.mkdir()
            (model / 'config.json').write_text(json.dumps(dict(model_type='qwen2', num_hidden_layers=48,
                hidden_size=64, num_attention_heads=4, num_key_value_heads=2)))
            (model / 'model.safetensors.index.json').write_text('{"weight_map":{"fake":"fake.safetensors"}}')
            (model / 'fake.safetensors').touch()  # Preflight fixture, never loaded as a model.
            argv = ['smoke_aime2025.py', '--methods', 'original', '--aime_protocol', 'thinkv_mixed',
                    '--aime_dataset_path', str(data), '--model_path', str(model),
                    '--output_dir', str(out), '--no-wandb']
            commands = []

            def fake_stage(cmd, *args, **kwargs):
                commands.append(cmd)
                selected = load_manifest(cmd[cmd.index('--aime_manifest') + 1])
                if cmd[2].endswith('evaluation.py'):
                    path = Path(cmd[cmd.index('--output_path') + 1])
                    path.parent.mkdir()
                    samples = load_problems(data, selected)
                    path.write_text(json.dumps({f"{s['id']}.0": dict(judgement=True, hit_token_limit=False,
                        output_len=3, input_len=2, gold=s['answer'], response_answer=s['answer'], response='synthetic')
                        for s in samples}))
                else:
                    from pm_kvq.evaluation.judge import judge_aime
                    self.assertEqual(judge_aime(out / 'original', aime_manifest=out / 'aime_manifest.json',
                                               n_responses=1)[:2], (100., 100.))
                return 0

            with patch.object(sys, 'argv', argv), patch.dict('os.environ'), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.get_device_name', return_value='CPU test fixture'), \
                    patch('torch.randn', return_value=torch.ones(2, 2)), \
                    patch('scripts.smoke_aime2025._tee_pty', side_effect=fake_stage):
                main()
            self.assertEqual(len(commands), 2)  # One evaluator and one judge.
            coverage = json.loads((out / 'coverage.json').read_text())
            metadata = json.loads((out / 'metadata.json').read_text())
            selected = load_manifest(out / 'aime_manifest.json')
            self.assertEqual((out / 'aime_manifest.json').read_bytes(), manifest_bytes(selected))
            self.assertEqual(coverage['expected_responses'], 2)
            self.assertEqual(coverage['methods'], {'original': {'responses': 2, 'complete': True}})
            self.assertEqual(coverage['problem_ids'], ['aime_2024_I.1', 'aime_2025_I.1'])
            self.assertEqual(metadata['aime']['manifest_sha256'], coverage['manifest_sha256'])
            self.assertEqual(metadata['status'], 'complete')
            self.assertNotIn('problem_indices', metadata)

    def test_thinkv_command(self):
        args = build_parser().parse_args(['--methods', 'original', 'thinkv', '--thinkv_calibration', '/cal.json',
                                          '--thinkv_token_budget', '2048', '--thinkv_reasoning_bits', '8',
                                          '--thinkv_refresh_interval', '64'])
        options = _method_options('thinkv', args, Path('/run'))
        self.assertEqual(options, ['--thinkv_calibration', Path('/run/thinkv_calibration.json'),
                                  '--thinkv_token_budget', 2048, '--thinkv_refresh_interval', 64,
                                  '--thinkv_reasoning_bits', 8])


if __name__ == '__main__':
    unittest.main()
