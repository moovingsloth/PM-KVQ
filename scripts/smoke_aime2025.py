"""Run the R1-Qwen-14B BF16/PM-KVQ smoke pipeline using the active Conda Python."""
import argparse
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
    child_env,
    init_smoke_run,
    log_smoke_outputs,
    log_stage,
    publish_existing_run,
    save_live,
    wandb_config_from_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


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


def _tee_pty(cmd, log, cwd, env):
    """Run cmd on a PTY so tqdm/HF bars render, and copy output to the log."""
    master, slave = pty.openpty()
    _set_winsize(slave)
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=slave, stderr=subprocess.STDOUT, close_fds=True)
    finally:
        os.close(slave)
    try:
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            sys.stdout.write(chunk.decode('utf-8', 'replace'))
            sys.stdout.flush()
            log.write(chunk)
            log.flush()
    finally:
        os.close(master)
    return proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', default='/home/dongwon/workspace/models/DeepSeek-R1-Distill-Qwen-14B')
    parser.add_argument('--dataset_root', default='/home/dongwon/workspace/datasets')
    parser.add_argument('--output_dir', default=str(ROOT / 'outputs' / 'smoke' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
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
    metadata = dict(model_path=str(model), dataset_root=str(data), torch=torch.__version__,
                    transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                    calibration_samples=8, seq_len=2048, effective_len=8192,
                    memory_budget_mb=budget, problem_indices=[0, 15], n_responses=1,
                    seed=42, max_new_tokens=32768, temperature=0.6, top_p=0.95,
                    backend='fake', status='running', stages=[])
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
            name=args.wandb_name or f'smoke-{out.name}',
            config=wandb_config_from_metadata(metadata),
        )
        print(f'W&B: {wandb_run.url}', flush=True)
        save_live(out / 'metadata.json', out)
    env = child_env(os.environ) if args.wandb else dict(os.environ, PYTHONUNBUFFERED='1')
    env.pop('TQDM_DISABLE', None)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), env.get('PYTHONPATH')]))
    n_stages = 10
    stage_i = 0

    def run(name, script, *options):
        nonlocal stage_i
        stage_i += 1
        cmd = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, options)]
        print(f'[{stage_i}/{n_stages} {100 * (stage_i - 1) / n_stages:.0f}%] {name}', flush=True)
        print(shlex.join(cmd), flush=True)
        start = time.monotonic()
        log_path = out / f'{name}.log'
        log_path.touch()
        if wandb_run is not None:
            save_live(log_path, out)
        with log_path.open('wb') as log:
            log.write((shlex.join(cmd) + '\n').encode())
            log.flush()
            returncode = _tee_pty(cmd, log, ROOT, env)
        elapsed = time.monotonic() - start
        metadata['stages'].append(dict(name=name, command=cmd, seconds=elapsed, returncode=returncode))
        if returncode:
            metadata['status'] = 'failed'
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(f'[{stage_i}/{n_stages} {100 * stage_i / n_stages:.0f}%] {name} finished in {elapsed:.1f}s', flush=True)
        if wandb_run is not None:
            log_stage(wandb_run, name, elapsed, returncode, stage_i)
        if returncode:
            raise RuntimeError(f'{name} failed; see {out / (name + ".log")}')

    try:
        common = ['--model_path', model, '--dataset_path', data / 'redpajama-1t-sample',
                  '--n_samples', 8, '--seq_len', 2048, '--effective_len', 8192]
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
        for method in ('original', 'pm-kvq'):
            for index in (0, 15):
                options = ['--model_path', model, '--dataset_path', data / 'aime', '--benchmark', 'aime',
                           '--version', 2025, '--start', index, '--end', index+1, '--n_responses', 1,
                           '--method', method, '--output_path', out / method / f'{index}.json']
                if method == 'pm-kvq':
                    options += ['--backend', 'fake', '--rep_scales', out / 'scales.pt', '--kv_budgets', out / 'budgets.pt']
                run(f'{method}_{index}', 'evaluation.py', *options)
            run(f'{method}_judge', 'judge.py', '--benchmark', 'aime', '--version', 2025, '--responses_dir', out / method)
        summary = {}
        for method in ('original', 'pm-kvq'):
            records = {}
            for index in (0, 15):
                records.update(json.loads((out / method / f'{index}.json').read_text()))
            assert set(records) == {'aime_2025_I.1.0', 'aime_2025_II.1.0'}
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
            log_smoke_outputs(wandb_run, out, metadata, summary)
            wandb_run.finish(exit_code=0)
        print(f'Completed smoke test: {out}', flush=True)
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
