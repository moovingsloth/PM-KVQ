import copy
import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pm_kvq.evaluation.aime_manifest import (
    MIXED_MANIFEST, expected_response_keys, load_manifest, load_problems,
    load_responses, manifest_bytes, provenance, select_mixed_manifest, validate_manifest,
)
from pm_kvq.evaluation.eval_aime import eval_aime
from pm_kvq.evaluation.judge import judge_aime
from pm_kvq.utils.wandb_logging import ARTIFACT_FILES, wandb_config_from_metadata


FIXED_IDS = [
    'aime_2024_I.1', 'aime_2024_I.2', 'aime_2024_I.3', 'aime_2024_I.4',
    'aime_2024_I.5', 'aime_2024_I.8', 'aime_2024_I.9', 'aime_2024_I.14',
    'aime_2024_II.3', 'aime_2024_II.4', 'aime_2024_II.6', 'aime_2024_II.9',
    'aime_2024_II.10', 'aime_2024_II.13', 'aime_2024_II.14',
    'aime_2025_I.1', 'aime_2025_I.3', 'aime_2025_I.7', 'aime_2025_I.8',
    'aime_2025_I.9', 'aime_2025_I.14', 'aime_2025_I.15', 'aime_2025_II.2',
    'aime_2025_II.3', 'aime_2025_II.5', 'aime_2025_II.6', 'aime_2025_II.9',
    'aime_2025_II.10', 'aime_2025_II.13', 'aime_2025_II.14',
]


def write_fixture(root, reverse_answers=True):
    """Synthetic local files only; never download problems or load a model."""
    for year in (2024, 2025):
        for subset in ('I', 'II'):
            directory = root / f'aime_{year}_{subset}'
            (directory / 'problems').mkdir(parents=True)
            indices = list(range(1, 16))
            answers = {}
            for i in indices:
                (directory / 'problems' / f'{i}.tex').write_text(f'Synthetic {year} {subset} {i}.')
                answers[i] = i + (100 if year == 2025 else 0) + (20 if subset == 'II' else 0)
            if reverse_answers:
                indices.reverse()
            (directory / 'answers.csv').write_text(
                'id,answer\n' + ''.join(f'{i},{answers[i]:03d}\n' for i in indices))


class ManifestTests(unittest.TestCase):
    def test_exact_selection_and_sampling_provenance(self):
        source = load_manifest(MIXED_MANIFEST, require_full=True)
        self.assertEqual(source, dict(schema_version=1, protocol='thinkv_mixed', seed=42,
                                      problem_ids=FIXED_IDS))
        rng = random.Random(42)
        sampled = []
        for year in (2024, 2025):
            pool = [f'aime_{year}_{subset}.{i}' for subset in ('I', 'II') for i in range(1, 16)]
            selected = rng.sample(pool, 15)
            self.assertEqual(len(set(selected)), 15)
            sampled += sorted(selected, key=lambda p: (p.split('.')[0], int(p.split('.')[1])))
        self.assertEqual(sampled, FIXED_IDS)
        self.assertEqual(len(set(FIXED_IDS)), 30)

    def test_preset_counts_and_no_runtime_sampling(self):
        with patch('random.Random', side_effect=AssertionError('runtime must not sample')):
            for preset, count, responses, total in [('smoke', 1, 1, 2), ('day', 5, 4, 40), ('full', 15, 16, 480)]:
                with self.subTest(preset=preset):
                    manifest = select_mixed_manifest(preset)
                    self.assertEqual(manifest['problem_ids'], FIXED_IDS[:count] + FIXED_IDS[15:15 + count])
                    self.assertEqual(len(expected_response_keys(manifest, responses)), total)

    def test_semantic_validation(self):
        source = load_manifest(MIXED_MANIFEST)
        for key, value in [('schema_version', True), ('schema_version', 2), ('seed', True),
                           ('seed', 41), ('protocol', 'unknown'), ('problem_ids', []),
                           ('problem_ids', 'aime_2024_I.1')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_manifest(dict(source, **{key: value}))
        for bad_id in ['aime_2024_I.0', 'aime_2024_I.16', 'aime_2024_III.1',
                       'aime_2024_I.01', 'aime_2023_I.1', '../aime_2024_I.1',
                       'aime_2025_I.1.0', 'aime_2025_I.1\n', 2024, ['aime_2024_I.1']]:
            invalid = copy.deepcopy(source)
            invalid['problem_ids'][0] = bad_id
            with self.subTest(bad_id=bad_id), self.assertRaisesRegex(ValueError, 'invalid AIME'):
                validate_manifest(invalid)
        invalid = copy.deepcopy(source)
        invalid['problem_ids'][0] = invalid['problem_ids'][1]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_manifest(invalid)
        with self.assertRaisesRegex(ValueError, 'year counts'):
            validate_manifest(dict(source, problem_ids=FIXED_IDS[:-1]), require_full=True)
        with self.assertRaisesRegex(ValueError, 'year counts'):
            validate_manifest(select_mixed_manifest('smoke'), require_full=True)
        wrong_protocol = dict(source, protocol='aime2025', problem_ids=[
            f'aime_2025_{subset}.{i}' for subset in ('I', 'II') for i in range(1, 16)])
        with patch('pm_kvq.evaluation.aime_manifest.load_manifest', return_value=wrong_protocol):
            with self.assertRaisesRegex(ValueError, 'mixed source manifest'):
                select_mixed_manifest('smoke')
        for n_responses in (None, True, 0, -1, 1.0):
            with self.subTest(n_responses=n_responses), self.assertRaises(ValueError):
                expected_response_keys(source, n_responses)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text('{"seed":42,"seed":42}')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
                load_manifest(path)


class LoaderAndJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        write_fixture(self.data)
        self.manifest = select_mixed_manifest('smoke')
        self.path = self.root / 'manifest.json'
        self.path.write_bytes(manifest_bytes(self.manifest))
        self.responses = self.root / 'responses'
        self.responses.mkdir()

    def test_year_identity_answer_mapping_and_manifest_order(self):
        reverse = dict(self.manifest, problem_ids=list(reversed(self.manifest['problem_ids'])))
        samples = load_problems(self.data, reverse)
        self.assertEqual([s['id'] for s in samples], ['aime_2025_I.1', 'aime_2024_I.1'])
        self.assertEqual([s['answer'] for s in samples], ['101', '1'])
        self.assertEqual([s['problem'] for s in samples], ['Synthetic 2025 I 1.', 'Synthetic 2024 I 1.'])

    def test_missing_or_invalid_local_data(self):
        problem = self.data / 'aime_2024_I' / 'problems' / '1.tex'
        problem.unlink()
        with self.assertRaises(FileNotFoundError):
            load_problems(self.data, self.manifest)
        problem.write_text('  \n')
        with self.assertRaisesRegex(ValueError, 'empty problem'):
            load_problems(self.data, self.manifest)
        problem.write_text('Synthetic problem')
        answers = self.data / 'aime_2024_I' / 'answers.csv'
        for contents, error in [('id,wrong\n1,001\n', 'answer column'),
                                ('id,answer\n2,002\n', 'missing answer'),
                                ('id,answer\n1,001\n1,002\n', 'duplicate answer'),
                                ('id,answer\n1,NaN\n', 'invalid AIME answer'),
                                ('id,answer\n16,001\n', 'invalid answer ID')]:
            answers.write_text(contents)
            with self.subTest(contents=contents), self.assertRaisesRegex(ValueError, error):
                load_problems(self.data, self.manifest)
        answers.unlink()
        with self.assertRaises(FileNotFoundError):
            load_problems(self.data, self.manifest)

    def test_content_hashes_and_wandb_configuration(self):
        samples = load_problems(self.data, self.manifest)
        metadata = provenance(self.data, self.manifest, samples)
        self.assertEqual(metadata['manifest_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(metadata['dataset_path'], str(self.data.resolve()))
        self.assertEqual(wandb_config_from_metadata({'aime': metadata})['aime'], metadata)
        self.assertIn('aime_manifest.json', ARTIFACT_FILES)
        self.assertIn('coverage.json', ARTIFACT_FILES)
        answers = self.data / 'aime_2024_I' / 'answers.csv'
        answers.write_text(answers.read_text().replace('1,001\n', '1,002\n'))
        changed = provenance(self.data, self.manifest, load_problems(self.data, self.manifest))
        self.assertNotEqual(metadata['selected_content_sha256'], changed['selected_content_sha256'])
        self.assertNotEqual(metadata['content_hashes']['aime_2024_I.1']['answer_sha256'],
                            changed['content_hashes']['aime_2024_I.1']['answer_sha256'])
        problem = self.data / 'aime_2025_I' / 'problems' / '1.tex'
        problem.write_text(problem.read_text() + ' Changed.')
        changed_again = provenance(self.data, self.manifest, load_problems(self.data, self.manifest))
        self.assertNotEqual(changed['selected_content_sha256'], changed_again['selected_content_sha256'])

    def write_responses(self):
        # 2024: 3/4 correct, vote correct. 2025: 1/4 correct, vote wrong.
        records = {}
        for problem, gold, answers in [('aime_2024_I.1', '1', ['1', '1', '1', '9']),
                                        ('aime_2025_I.1', '101', ['9', '9', '9', '101'])]:
            for i, answer in enumerate(answers):
                records[f'{problem}.{i}'] = dict(response='synthetic', response_answer=answer,
                    gold=gold, judgement=answer == gold, input_len=10, output_len=20 + i)
        (self.responses / 'responses.json').write_text(json.dumps(records))
        return records

    def test_judge_metrics_and_order_with_year_collisions(self):
        self.write_responses()
        results = judge_aime(self.responses, version=9999, aime_manifest=self.path, n_responses=4)
        self.assertEqual(results, (50., 50., [75., 25.], [True, False], [31.5, 31.5]))
        reversed_manifest = dict(self.manifest, problem_ids=list(reversed(self.manifest['problem_ids'])))
        self.path.write_bytes(manifest_bytes(reversed_manifest))
        results = judge_aime(self.responses, aime_manifest=self.path, n_responses=4)
        self.assertEqual(results[2], [25., 75.])
        with self.assertRaisesRegex(ValueError, 'explicit positive'):
            judge_aime(self.responses, aime_manifest=self.path)

    def test_response_missing_duplicate_and_unexpected_rejected(self):
        records = self.write_responses()
        path = self.responses / 'responses.json'
        key = next(iter(records))
        invalid = dict(records)
        invalid.pop(key)
        path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'missing='):
            judge_aime(self.responses, aime_manifest=self.path, n_responses=4)
        invalid = dict(records, **{'aime_2024_I.2.0': records[key]})
        path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'unexpected='):
            load_responses(self.responses, self.manifest, 4)
        path.write_text(json.dumps(records))
        duplicate = self.responses / 'duplicate.json'
        duplicate.write_text(json.dumps({key: records[key]}))
        with self.assertRaisesRegex(ValueError, 'duplicate response'):
            judge_aime(self.responses, aime_manifest=self.path, n_responses=4)
        duplicate.unlink()
        path.write_text('{"same":{},"same":{}}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            load_responses(self.responses, self.manifest, 4)

    def test_judge_rejects_invalid_records_and_inconsistent_gold(self):
        records = self.write_responses()
        key = next(iter(records))
        for field, value in [('judgement', 1), ('gold', ''), ('input_len', -1),
                             ('output_len', True), ('response_answer', 1)]:
            invalid = copy.deepcopy(records)
            invalid[key][field] = value
            (self.responses / 'responses.json').write_text(json.dumps(invalid))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'invalid response'):
                judge_aime(self.responses, aime_manifest=self.path, n_responses=4)
        records[key]['gold'] = '2'
        (self.responses / 'responses.json').write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, 'inconsistent gold'):
            judge_aime(self.responses, aime_manifest=self.path, n_responses=4)

    def test_evaluator_uses_manifest_order_and_response_seeds(self):
        output = self.responses / 'responses.json'
        with patch('pm_kvq.evaluation.eval_aime.chat',
                   side_effect=[('\\boxed{1}', [10, 20]), ('\\boxed{9}', [10, 22]),
                                ('\\boxed{101}', [10, 24]), ('\\boxed{101}', [10, 26])]) as chat:
            accuracy = eval_aime(SimpleNamespace(), None, dataset_path=self.data, aime_manifest=self.path,
                                 version=9999, start=99, end=100, n_responses=2, output_path=str(output), seed=42)
        self.assertEqual(accuracy, .75)
        records = load_responses(self.responses, self.manifest, 2)
        self.assertEqual(list(records), expected_response_keys(self.manifest, 2))
        self.assertEqual([r['seed'] for r in records.values()], [42, 43, 42, 43])
        self.assertEqual(chat.call_count, 4)
        self.assertIn('Synthetic 2024 I 1.', chat.call_args_list[0].kwargs['text'])
        self.assertIn('Synthetic 2025 I 1.', chat.call_args_list[2].kwargs['text'])
        self.assertEqual(judge_aime(self.responses, aime_manifest=self.path, n_responses=2)[:2], (75., 100.))

    def test_legacy_evaluator_and_judge_remain_available(self):
        # The legacy CSV path uses row order, so provide ordered legacy CSVs.
        for subset in ('I', 'II'):
            (self.data / f'aime_2025_{subset}' / 'answers.csv').write_text(
                'id,answer\n' + ''.join(f'{i},{i:03d}\n' for i in range(1, 16)))
        output = self.responses / 'responses.json'
        with patch('pm_kvq.evaluation.eval_aime.chat', return_value=('\\boxed{1}', [10, 20])):
            self.assertEqual(eval_aime(SimpleNamespace(), None, dataset_path=self.data,
                version=2025, start=0, end=1, n_responses=1, output_path=str(output)), 1.)
        self.assertEqual(list(json.loads(output.read_text())), ['aime_2025_I.1.0'])
        self.assertEqual(judge_aime(self.responses, version=2025)[:2], (100., 100.))

    def test_evaluation_cli_preflight_before_model_loading(self):
        (self.data / 'aime_2024_I' / 'problems' / '1.tex').unlink()
        argv = ['scripts/evaluation.py', '--method', 'original', '--aime_manifest', str(self.path),
                '--dataset_path', str(self.data), '--model_path', '/must-not-load']
        with patch.object(sys, 'argv', argv), patch('transformers.AutoModelForCausalLM.from_pretrained') as load:
            with self.assertRaises(FileNotFoundError):
                runpy.run_path('scripts/evaluation.py', run_name='__main__')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
