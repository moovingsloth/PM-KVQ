"""Run the R1-Qwen-14B BF16/PM-KVQ pipeline, with opt-in ThinKV, using the active Conda Python."""
import argparse
import codecs
import datetime
import fcntl
import json
import os
from pathlib import Path
import pty
import shlex
import struct
import subprocess
import sys
import termios
import time

from pm_kvq.evaluation.aime_manifest import (
    expected_response_keys,
    load_problems,
    load_responses,
    manifest_bytes,
    provenance,
    select_mixed_manifest,
    validate_manifest,
)
from pm_kvq.utils.wandb_logging import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    PROJECT_URL,
    ConsoleCollapser,
    child_env,
    copy_into_run,
    emit_console,
    init_smoke_run,
    make_console_emitter,
    log_smoke_outputs,
    log_stage,
    publish_existing_run,
    wandb_config_from_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    'smoke': {'n_samples': 8, 'n_responses': 1, 'slices': [(0, 1), (15, 16)]},
    'day': {'n_samples': 8, 'n_responses': 4, 'slices': [(0, 5), (15, 20)]},
    'full': {'n_samples': 512, 'n_responses': 16, 'slices': [(0, 30)]},
}


def _problem_id(index):
    if index < 15:
        return 'I', index + 1
    return 'II', index - 14


def _expected_keys(slices, n_responses):
    return {
        f'aime_2025_{subset}.{problem}.{response}'
        for start, end in slices
        for index in range(start, end)
        for subset, problem in [_problem_id(index)]
        for response in range(n_responses)
    }


def _eval_jobs(method, slices, out, aime_manifest=None):
    if aime_manifest is not None:
        return [(f'{method}_evaluation', 0, len(aime_manifest['problem_ids']), out / method / 'responses.json')]
    jobs = []
    for start, end in slices:
        if end - start == 1:
            name = f'{method}_{start}'
            path = out / method / f'{start}.json'
        elif slices == [(0, 30)]:
            name = f'{method}_evaluation'
            path = out / method / 'responses.json'
        else:
            name = f'{method}_{start}_{end}'
            path = out / method / f'{start}_{end}.json'
        jobs.append((name, start, end, path))
    return jobs


def _aime_selection(args):
    if args.aime_protocol == 'thinkv_mixed':
        manifest = select_mixed_manifest(args.preset)
        default_path = ROOT / 'datasets' / 'aime'
    else:
        manifest = validate_manifest(dict(schema_version=1, protocol='aime2025', seed=42,
            problem_ids=[f'aime_2025_{subset}.{problem}'
                         for start, end in PRESETS[args.preset]['slices']
                         for index in range(start, end)
                         for subset, problem in [_problem_id(index)]]))
        default_path = Path(args.dataset_root) / 'aime'
    return manifest, Path(args.aime_dataset_path or default_path).resolve()


def _evaluation_options(method, args, out, model, aime_data, start, end, output_path):
    options = ['--model_path', model, '--dataset_path', aime_data, '--benchmark', 'aime',
               '--n_responses', PRESETS[args.preset]['n_responses'], '--seed', 42,
               '--method', method, '--output_path', output_path]
    if args.aime_protocol == 'thinkv_mixed':
        options += ['--aime_manifest', out / 'aime_manifest.json']
    else:
        options += ['--version', 2025, '--start', start, '--end', end]
    return options + _method_options(method, args, out)


def _judge_options(method, args, out):
    options = ['--benchmark', 'aime', '--responses_dir', out / method]
    if args.aime_protocol == 'thinkv_mixed':
        options += ['--aime_manifest', out / 'aime_manifest.json',
                    '--n_responses', PRESETS[args.preset]['n_responses']]
    else:
        options += ['--version', 2025]
    return options


def _set_winsize(fd):
    size = struct.pack('HHHH', 24, 120, 0, 0)
    if sys.stdout.isatty():
        try:
            size = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, size)
        except OSError:
            pass
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    except OSError:
        pass


def _tee_pty(cmd, log, cwd, env, emit=emit_console):
    """Run cmd on a PTY so tqdm/HF bars render, and copy output to the log.

    Collapsed text is emitted to the terminal and to W&B Logs while the child runs.
    """
    master, slave = pty.openpty()
    _set_winsize(slave)
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=slave, stderr=subprocess.STDOUT, close_fds=True)
    finally:
        os.close(slave)
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    collapser = ConsoleCollapser(emit)
    try:
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            text = decoder.decode(chunk)
            if text:
                collapser.feed(text, live=True)
        tail = decoder.decode(b'', final=True)
        if tail:
            collapser.feed(tail, live=True)
        collapser.flush()
    finally:
        os.close(master)
    return proc.wait()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='smoke',
                        help='smoke: 8 calib samples, AIME I.1 and II.1, 1 response. '
                             'day: 8 calib samples, 10 problems × 4 responses (~0.5–1 day). '
                             'full: 512 calib samples, 30 problems × 16 responses')
    parser.add_argument('--model_path', default='/home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B')
    parser.add_argument('--dataset_root', default='/home/dongwon/workspace/datasets')
    parser.add_argument('--aime_protocol', choices=['aime2025', 'thinkv_mixed'], default='aime2025',
                        help='thinkv_mixed: fixed seed-42 selection, 1/5/15 problems per year for smoke/day/full')
    parser.add_argument('--aime_dataset_path', default=None,
                        help='Local AIME root; mixed defaults to repository datasets/aime, aime2025 to dataset_root/aime')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--methods', nargs='+', choices=['original', 'pm-kvq', 'thinkv'],
                        default=['original', 'pm-kvq'], help='Opt-in method selection; existing pair is the default')
    parser.add_argument('--thinkv_calibration', default=None)
    parser.add_argument('--thinkv_token_budget', type=int, default=1024)
    parser.add_argument('--thinkv_refresh_interval', type=int, default=None)
    parser.add_argument('--thinkv_reasoning_bits', type=int, choices=[4, 8], default=None)
    parser.add_argument('--thinkv_execution_bits', type=int, choices=[4], default=None)
    parser.add_argument('--thinkv_transition_bits', type=int, choices=[2], default=None)
    parser.add_argument('--wandb', dest='wandb', action='store_true', default=True,
                        help=f'Log this run to {PROJECT_URL} (default)')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_entity', default=os.environ.get('WANDB_ENTITY', DEFAULT_ENTITY))
    parser.add_argument('--wandb_project', default=os.environ.get('WANDB_PROJECT', DEFAULT_PROJECT))
    parser.add_argument('--wandb_name', default=os.environ.get('WANDB_NAME'))
    parser.add_argument('--log_existing', default=None,
                        help='Publish an already-finished output directory to W&B without rerunning')
    return parser


def _method_options(method, args, out):
    if method == 'pm-kvq':
        return ['--backend', 'fake', '--rep_scales', out / 'scales.pt',
                '--kv_budgets', out / 'budgets.pt', '--n_sink_token', 1,
                '--n_sink_token_bits', 16, '--n_window_token', 128,
                '--n_window_token_bits', 16, '--n_init_kv_bits', 16]
    if method == 'thinkv':
        options = ['--thinkv_calibration', out / 'thinkv_calibration.json',
                   '--thinkv_token_budget', args.thinkv_token_budget]
        for name in ('refresh_interval', 'reasoning_bits', 'execution_bits', 'transition_bits'):
            value = getattr(args, f'thinkv_{name}')
            if value is not None:
                options += [f'--thinkv_{name}', value]
        return options
    return []


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error('--methods must not contain duplicates')
    thinkv_artifact = None
    if 'thinkv' in args.methods and not args.log_existing:
        if args.thinkv_calibration is None:
            parser.error('--methods thinkv requires --thinkv_calibration')
        from dataclasses import replace
        from pm_kvq.quantization.methods.thinkv.config import load_calibration, numerical_settings
        thinkv_artifact, thinkv_config = load_calibration(args.thinkv_calibration)
        overrides = {name: getattr(args, f'thinkv_{name}') for name in
                     ('refresh_interval', 'reasoning_bits', 'execution_bits', 'transition_bits')
                     if getattr(args, f'thinkv_{name}') is not None}
        thinkv_config = replace(thinkv_config, token_budget=args.thinkv_token_budget, **overrides)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.output_dir is None:
        args.output_dir = str(ROOT / 'outputs' / args.preset / stamp)
    if args.log_existing:
        url = publish_existing_run(
            Path(args.log_existing),
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
        )
        print(f'Published existing smoke run to {url}', flush=True)
        return
    selection, aime_data = _aime_selection(args)
    samples = load_problems(aime_data, selection)
    aime_metadata = provenance(aime_data, selection, samples)
    if args.aime_protocol == 'aime2025':
        # The preserved legacy evaluator reads both complete contests before slicing.
        full_selection = dict(selection, problem_ids=[f'aime_2025_{subset}.{i}'
                              for subset in ('I', 'II') for i in range(1, 16)])
        load_problems(aime_data, full_selection)
    os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
    os.environ['HF_XET_CACHE'] = str(ROOT / '.cache' / 'huggingface' / 'xet')
    os.environ['HF_DATASETS_CACHE'] = str(ROOT / '.cache' / 'huggingface' / 'datasets')
    import torch
    import transformers
    assert torch.cuda.is_available(), 'Active Conda environment requires CUDA-enabled PyTorch'
    x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all().item(), 'CUDA computation failed'
    model = Path(args.model_path).resolve()
    data = Path(args.dataset_root).resolve()
    config = json.loads((model / 'config.json').read_text())
    if thinkv_artifact is not None:
        from pm_kvq.quantization.methods.thinkv.config import model_identity, validate_calibration
        validate_calibration(thinkv_artifact, model_identity(
            transformers.AutoConfig.from_pretrained(str(model), local_files_only=True), str(model)))
    assert config['model_type'] == 'qwen2' and config['num_hidden_layers'] == 48
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        assert (model / shard).is_file(), f'Model download incomplete: {shard}'
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'aime_manifest.json').write_bytes(manifest_bytes(selection))
    if thinkv_artifact is not None:
        (out / 'thinkv_calibration.json').write_text(json.dumps(thinkv_artifact, indent=2, allow_nan=False))
    kv_dim = config['num_key_value_heads'] * config['hidden_size'] // config['num_attention_heads']
    # Paper Qwen-14B mixed 2/4 row on A100-40G: 2-bit@32k * (BS16/BS12) = 1024 MiB/request.
    # GB10 leftover ~94 GiB does not change this per-request budget.
    budget = config['num_hidden_layers'] * kv_dim * 2 * 32768 * 2 / 8 / 2**20 * 16 / 12
    preset = PRESETS[args.preset]
    n_samples = preset['n_samples']
    n_responses = preset['n_responses']
    slices = preset['slices']
    problem_indices = [index for start, end in slices for index in range(start, end)]
    metadata = dict(preset=args.preset, model_path=str(model), dataset_root=str(data),
                    torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(0), calibration_samples=n_samples,
                    seq_len=2048, effective_len=8192, memory_budget_mb=budget,
                    problem_indices=problem_indices, n_responses=n_responses,
                    seed=42, max_new_tokens=32768, temperature=0.6, top_p=0.95,
                    backend='fake', status='running', stages=[])
    metadata['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata['aime'] = aime_metadata
    if args.aime_protocol == 'thinkv_mixed':
        metadata.pop('problem_indices')  # Legacy 2025 slice indices do not identify mixed problems.
    metadata['methods'] = args.methods
    if 'pm-kvq' not in args.methods:
        metadata.update(calibration_samples=0, memory_budget_mb=None, backend=None)
    if thinkv_artifact is not None:
        metadata['thinkv'] = dict(calibration=thinkv_artifact, token_budget=args.thinkv_token_budget,
                                  numerical=numerical_settings(thinkv_config), storage='input_dtype_qdq_reference')
    model_metadata = model / '.cache' / 'huggingface' / 'download' / 'config.json.metadata'
    if model_metadata.is_file():
        metadata['model_revision'] = model_metadata.read_text().splitlines()[0]
    metadata['model_config'] = config
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Artifacts: {out}', flush=True)
    wandb_run = None
    if args.wandb:
        wandb_run = init_smoke_run(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name or f'{args.preset}-{out.name}',
            config=wandb_config_from_metadata(metadata),
            job_type=args.preset,
        )
        print(f'W&B: {wandb_run.url}', flush=True)
        copy_into_run(wandb_run, out / 'metadata.json')
        copy_into_run(wandb_run, out / 'aime_manifest.json')
    emit = make_console_emitter(wandb_run)
    env = child_env(os.environ) if args.wandb else dict(os.environ, PYTHONUNBUFFERED='1')
    env.pop('TQDM_DISABLE', None)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), env.get('PYTHONPATH')]))
    evaluation_manifest = selection if args.aime_protocol == 'thinkv_mixed' else None
    jobs_per_method = 1 if evaluation_manifest is not None else len(slices)
    n_stages = (4 if 'pm-kvq' in args.methods else 0) + len(args.methods) * (jobs_per_method + 1)
    stage_i = 0

    def run(name, script, *options):
        nonlocal stage_i
        stage_i += 1
        cmd = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, options)]
        print(f'[{stage_i}/{n_stages} {100 * (stage_i - 1) / n_stages:.0f}%] {name}', flush=True)
        print(shlex.join(cmd), flush=True)
        start = time.monotonic()
        log_path = out / f'{name}.log'
        with log_path.open('wb') as log:
            log.write((shlex.join(cmd) + '\n').encode())
            log.flush()
            returncode = _tee_pty(cmd, log, ROOT, env, emit=emit)
        elapsed = time.monotonic() - start
        metadata['stages'].append(dict(name=name, command=cmd, seconds=elapsed, returncode=returncode))
        if returncode:
            metadata['status'] = 'failed'
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(f'[{stage_i}/{n_stages} {100 * stage_i / n_stages:.0f}%] {name} finished in {elapsed:.1f}s', flush=True)
        if wandb_run is not None:
            copy_into_run(wandb_run, log_path, relative=f'stage_logs/{log_path.name}')
            copy_into_run(wandb_run, out / 'metadata.json')
            log_stage(wandb_run, name, elapsed, returncode, stage_i)
        if returncode:
            raise RuntimeError(f'{name} failed; see {out / (name + ".log")}')

    try:
        if 'pm-kvq' in args.methods:
            common = ['--model_path', model, '--dataset_path', data / 'redpajama-1t-sample',
                      '--n_samples', n_samples, '--seq_len', 2048, '--effective_len', 8192]
            run('sensitivity', 'get_sensitivity.py', *common, '--save_path', out / 'sensitivity.pt')
            run('allocation', 'allocate_memory.py', '--sensitivity_path', out / 'sensitivity.pt',
                '--memory_budget', budget, '--fbit_choices', '4,2', '--hidden_size', kv_dim,
                '--max_len', 32768, '--save_path', out / 'budgets.pt')
            run('max_keys', 'get_max_keys.py', *common, '--save_path', out / 'max_keys.pt')
            run('scales', 'search_rep_scales.py', *common, '--max_keys_path', out / 'max_keys.pt',
                '--k_bits', 2, '--v_bits', 2, '--save_path', out / 'scales.pt')
            budgets = torch.load(out / 'budgets.pt', map_location='cpu')
            scales = torch.load(out / 'scales.pt', map_location='cpu')
            assert len(budgets) == len(scales) == config['num_hidden_layers']
            assert all(b > 0 for b in budgets) and sum(budgets) <= budget + 1e-6
            assert all(torch.isfinite(s).all().item() and (s > 0).all().item() for s in scales)
        for method in args.methods:
            for name, eval_start, eval_end, output_path in _eval_jobs(method, slices, out, evaluation_manifest):
                options = _evaluation_options(method, args, out, model, aime_data, eval_start, eval_end, output_path)
                run(name, 'evaluation.py', *options)
            run(f'{method}_judge', 'judge.py', *_judge_options(method, args, out))
        expected = expected_response_keys(selection, n_responses)
        summary = {}
        coverage = dict(aime_metadata, n_responses=n_responses, expected_responses=len(expected), methods={})
        for method in args.methods:
            records = load_responses(out / method, selection, n_responses)
            coverage['methods'][method] = dict(responses=len(records), complete=True)
            summary[method] = {
                'correct': sum(r['judgement'] for r in records.values()), 'responses': len(records),
                'token_limit_hits': sum(r['hit_token_limit'] for r in records.values()),
                'output_tokens': {key: r['output_len'] for key, r in records.items()},
                'quantization_observed': any(int(bit) < 16 and count > 0 for r in records.values()
                    for layer in r.get('kv_bit_counts_by_layer', []) for bit, count in layer.items()),
            }
            if method == 'thinkv':
                summary[method]['quantization_observed'] = any(
                    count > 0 for r in records.values()
                    for layer in r['thinkv']['quantized_token_counts_by_layer'] for count in layer.values())
                summary[method]['eviction_observed'] = any(
                    count > 0 for r in records.values() for count in r['thinkv']['evicted_tokens_by_layer'])
                summary[method]['cache_tensor_bytes'] = {key: r['thinkv']['cache_tensor_bytes'] for key, r in records.items()}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        (out / 'coverage.json').write_text(json.dumps(coverage, indent=2))
        metadata['status'] = 'complete'
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        if wandb_run is not None:
            copy_into_run(wandb_run, out / 'summary.json')
            copy_into_run(wandb_run, out / 'coverage.json')
            copy_into_run(wandb_run, out / 'metadata.json')
            log_smoke_outputs(wandb_run, out, metadata, summary)
            wandb_run.finish(exit_code=0)
        print(f'Completed {args.preset} run: {out}', flush=True)
        if wandb_run is not None:
            print(f'W&B: {wandb_run.url}', flush=True)
    except BaseException:
        if wandb_run is not None:
            try:
                log_smoke_outputs(wandb_run, out, metadata)
            finally:
                wandb_run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
