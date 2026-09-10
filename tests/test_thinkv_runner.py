from pathlib import Path
import unittest

from scripts.smoke_aime2025 import PRESETS, _expected_keys, _method_options, build_parser


class RunnerTests(unittest.TestCase):
    def test_preserved_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.methods, ['original', 'pm-kvq'])
        self.assertEqual(args.preset, 'smoke')
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
