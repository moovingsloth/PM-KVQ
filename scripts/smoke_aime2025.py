"""Run the R1-Qwen-14B BF16/PM-KVQ smoke pipeline using the active Conda Python."""
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


def _slice_arg(slices):
    return ','.join(f'{start}:{end}' for start, end in slices)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='smoke',
                        help='smoke: 8 calib samples, AIME I.1 and II.1, 1 response. '
                             'day: 8 calib samples, 10 problems × 4 responses. '
                             'full: 512 calib samples, 30 problems × 16 responses. '
                             'This Spark branch loads calibration weights once and batches original responses.')
    parser.add_argument('--model_path', default='/home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B')
    parser.add_argument('--dataset_root', default='/home/dongwon/workspace/datasets')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--wandb', dest='wandb', action='store_true', default=True,
                        help=f'Log this run to {PROJECT_URL} (default)')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_entity', default=os.environ.get('WANDB_ENTITY', DEFAULT_ENTITY))
    parser.add_argument('--wandb_project', default=os.environ.get('WANDB_PROJECT', DEFAULT_PROJECT))
    parser.add_argument('--wandb_name', default=os.environ.get('WANDB_NAME'))
    parser.add_argument('--log_existing', default=None,
                        help='Publish an already-finished output directory to W&B without rerunning')
    args = parser.parse_args()
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
    assert config['model_type'] == 'qwen2' and config['num_hidden_layers'] == 48
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        assert (model / shard).is_file(), f'Model download incomplete: {shard}'
    for subset in ('I', 'II'):
        assert (data / 'aime' / f'aime_2025_{subset}' / 'problems' / '1.tex').is_file()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
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
                    gpu=torch.cuda.get_device_name(0), host='NVIDIA DGX Spark',
                    calibration_samples=n_samples,
                    seq_len=2048, effective_len=8192, memory_budget_mb=budget,
                    problem_indices=problem_indices, n_responses=n_responses,
                    seed=42, max_new_tokens=32768, temperature=0.6, top_p=0.95,
                    backend='fake', original_response_batch_size=4,
                    status='running', stages=[])
    metadata['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
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
    emit = make_console_emitter(wandb_run)
    env = child_env(os.environ) if args.wandb else dict(os.environ, PYTHONUNBUFFERED='1')
    env.pop('TQDM_DISABLE', None)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), env.get('PYTHONPATH')]))
    n_stages = 5
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
        common = ['--model_path', model, '--dataset_path', data / 'redpajama-1t-sample',
                  '--n_samples', n_samples, '--seq_len', 2048, '--effective_len', 8192]
        run('calibration', 'calibrate_pmkvq.py', *common,
            '--sensitivity_path', out / 'sensitivity.pt',
            '--max_keys_path', out / 'max_keys.pt',
            '--scales_path', out / 'scales.pt',
            '--k_bits', 2, '--v_bits', 2,
            '--memory_budget', budget, '--fbit_choices', '4,2',
            '--hidden_size', kv_dim, '--max_len', 32768,
            '--budgets_path', out / 'budgets.pt')
        budgets = torch.load(out / 'budgets.pt', map_location='cpu')
        scales = torch.load(out / 'scales.pt', map_location='cpu')
        assert len(budgets) == len(scales) == config['num_hidden_layers']
        assert all(b > 0 for b in budgets) and sum(budgets) <= budget + 1e-6
        assert all(torch.isfinite(s).all().item() and (s > 0).all().item() for s in scales)
        slice_arg = _slice_arg(slices)
        for method in ('original', 'pm-kvq'):
            output_path = out / method / 'responses.json'
            options = ['--model_path', model, '--dataset_path', data / 'aime', '--benchmark', 'aime',
                       '--version', 2025, '--slices', slice_arg,
                       '--n_responses', n_responses, '--seed', 42, '--method', method,
                       '--output_path', output_path,
                       '--response_batch_size', 4 if method == 'original' else 1]
            if method == 'pm-kvq':
                options += ['--backend', 'fake', '--rep_scales', out / 'scales.pt',
                            '--kv_budgets', out / 'budgets.pt', '--n_sink_token', 1,
                            '--n_sink_token_bits', 16, '--n_window_token', 128,
                            '--n_window_token_bits', 16, '--n_init_kv_bits', 16]
            run(f'{method}_evaluation', 'evaluation.py', *options)
            run(f'{method}_judge', 'judge.py', '--benchmark', 'aime', '--version', 2025, '--responses_dir', out / method)
        expected = _expected_keys(slices, n_responses)
        summary = {}
        for method in ('original', 'pm-kvq'):
            records = {}
            for path in sorted((out / method).glob('*.json')):
                records.update(json.loads(path.read_text()))
            assert set(records) == expected, f'{method} coverage {len(records)}/{len(expected)}'
            summary[method] = {
                'correct': sum(r['judgement'] for r in records.values()), 'responses': len(records),
                'token_limit_hits': sum(r['hit_token_limit'] for r in records.values()),
                'output_tokens': {key: r['output_len'] for key, r in records.items()},
                'quantization_observed': any(int(bit) < 16 and count > 0 for r in records.values()
                    for layer in r.get('kv_bit_counts_by_layer', []) for bit, count in layer.items()),
            }
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        metadata['status'] = 'complete'
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        if wandb_run is not None:
            copy_into_run(wandb_run, out / 'summary.json')
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
