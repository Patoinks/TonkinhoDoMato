"""Isolated CPU update experiments against a frozen source/checkpoint snapshot."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'perf_results/network-20260923'
MODES = ('gru', 'cache', 'target', 'optimizer', 'reply', 'buffers')


def reserved():
    # Agents reserve outputs in WORKING.md before invoking this script. Reread
    # the board before writes; don't require one agent's name for human reruns.
    (ROOT/'WORKING.md').read_text(encoding='utf-8')
    WORK.resolve().relative_to((ROOT/'perf_results').resolve())


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('Frozen source does not match patch: ' + before[:100])
    return source.replace(before, after)


def prepare(mode):
    reserved()
    destination = WORK/'variants'/mode
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WORK/'baseline/tonkai', destination/'tonkai', dirs_exist_ok=True)
    shutil.copy2(ROOT/'tonkai/network_fastpaths.py', destination/'tonkai/network_fastpaths.py')
    path = destination/'tonkai/survival_trainer.py'
    source = path.read_text(encoding='utf-8')
    changes = mode.split('+')
    source = 'from .network_fastpaths import gru_step, bootstrap_values, reply_losses, cached_future, unpack_reusing\n' + source
    if 'gru' in changes:
        source = replace_once(source, '        for t in range(stop):',
            '        with torch.set_grad_enabled(gradients):\n            input_gates = F.linear(latent, net.memory_cell.weight_ih, net.memory_cell.bias_ih)\n        for t in range(stop):')
        source = replace_once(source, 'candidate = net.memory_cell(features, memory)',
                              'candidate = gru_step(net.memory_cell, input_gates[:, t], memory)')
    if 'cache' in changes:
        source = replace_once(source,
            'F.normalize(self.future_encoder(x.flatten(0, 1)), dim=-1).reshape(b, total, -1)',
            'cached_future(self, batch, x)')
    if 'target' in changes:
        source = replace_once(source,
            '            target_values = self.target.survival(target_context.flatten(0, 1))[0].mean(-1).reshape(b, total)\n', '')
        source = replace_once(source, 'bootstrap = target_values.gather(1, endpoints)',
                              'bootstrap = bootstrap_values(self.target, target_context, endpoints)')
    if 'optimizer' in changes or 'fused' in changes:
        # load_state_dict restores the old flag, so set it at each update too.
        flags = ('            group["foreach"] = False\n            group["fused"] = True' if 'fused' in changes
                 else '            group["foreach"] = True')
        source = replace_once(source, '        self.optimizer.step()',
            '        for group in self.optimizer.param_groups:\n'+flags+'\n        self.optimizer.step()')
        source = replace_once(source,
            '            for target, current in zip(self.target.parameters(), self.net.parameters()):\n                target.lerp_(current, cfg.target_tau)',
            '            torch._foreach_lerp_(list(self.target.parameters()), list(self.net.parameters()), cfg.target_tau)')
    if 'reply' in changes:
        source = replace_once(source,
            '        logits = self.net.predict_logits(context).split(sizes)\n        losses = torch.stack([F.cross_entropy(logit[None], torch.tensor([label], device=self.device))\n                              for logit, label in zip(logits, labels)])',
            '        logits = self.net.predict_logits(context)\n        losses = reply_losses(logits, sizes, labels)')
    if 'buffers' in changes:
        source = replace_once(source, 'unpacked = unpack_many(batch["packed"].reshape(-1, batch["packed"].shape[-1]))',
                              'unpacked = unpack_reusing(self, batch["packed"].reshape(-1, batch["packed"].shape[-1]))')
    reserved()
    path.write_text(source, encoding='utf-8')
    return destination


def compact(state):
    # Optimizer execution flags are metadata, not learned moments or parameters.
    for group in state['optimizer']['param_groups']:
        group['foreach'] = None
        group['fused'] = None
    for episode in state['replay']['episodes']:
        for key in list(episode):
            if key not in ('id', 'priorities'):
                del episode[key]
    return state


def differences(a, b):
    import numpy as np
    import torch
    report = {'exact': True, 'max_abs': {}, 'different_scalars': 0}
    def visit(x, y, path):
        if isinstance(x, torch.Tensor):
            x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            same = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
            if not same:
                report['exact'] = False
                group = '.'.join(path.split('.')[:3])
                delta = float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))
                report['max_abs'][group] = max(report['max_abs'].get(group, 0), delta)
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for key in x:
                visit(x[key], y[key], path+'.'+str(key))
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), path
            for index, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, path+'.'+str(index))
        elif dataclasses.is_dataclass(x):
            for field in dataclasses.fields(x):
                visit(getattr(x, field.name), getattr(y, field.name), path+'.'+field.name)
        elif x != y:
            report['exact'] = False
            report['different_scalars'] += 1
    visit(a, b, 'state')
    return report


def child(args):
    sys.path.insert(0, str(WORK/'variants'/args.child))
    sys.path.append(str(ROOT))
    import torch
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_trainer import CognitiveTrainer
    torch.set_num_threads(1)
    state = torch.load(WORK/'input.pt', map_location='cpu', weights_only=False)
    cfg = CognitiveConfig.from_dict(state['config'])
    cfg.device = 'cpu'
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(state)
    # Same updates/RNG in every mode. Warmup also exercises stateful caches.
    for _ in range(10):
        trainer.update()
    losses = []
    start = time.perf_counter()
    for _ in range(args.updates):
        losses.append(trainer.update())
    seconds = time.perf_counter()-start
    evidence = {'trainer': compact(trainer.state_dict()), 'losses': losses}
    reference = WORK/f'reference-{args.updates}.pt'
    if args.child == 'baseline' and not reference.exists():
        reserved()
        torch.save(evidence, reference)
    diff = differences(torch.load(reference, weights_only=False, map_location='cpu'), evidence)
    row = dict(mode=args.child, updates=args.updates, seconds=seconds, updates_per_second=args.updates/seconds,
               equivalence=diff, cache_stats=getattr(trainer, '_future_cache_stats', None))
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--modes', nargs='+', default=list(MODES))
    parser.add_argument('--output', default='results.json')
    args = parser.parse_args()
    if args.updates < 1 or any(part not in (*MODES, 'fused') for mode in args.modes for part in mode.split('+')):
        parser.error('Use positive updates and known experiment modes')
    if Path(args.output).name != args.output:
        parser.error('Output must be a filename inside the benchmark directory')
    if args.child:
        child(args)
        return
    for mode in ('baseline', *args.modes):
        prepare(mode)
    rows = []
    order = ['baseline', *args.modes, *reversed(args.modes), 'baseline']
    for mode in order:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', mode,
                                 '--updates', str(args.updates)], text=True, capture_output=True, cwd=ROOT)
        if result.returncode:
            raise RuntimeError(result.stderr)
        row = json.loads(result.stdout)
        rows.append(row)
        print(json.dumps(row), flush=True)
    manifest = json.loads((WORK/'checkpoint.json').read_text())
    means = {mode: sum(r['seconds'] for r in rows if r['mode']==mode)/2 for mode in order}
    report = dict(runs=rows, mean_seconds=means,
                  speedups={mode:means['baseline']/value for mode,value in means.items()},
                  source_checkpoint_unchanged=hashlib.sha256(Path(manifest['source_checkpoint']).read_bytes()).hexdigest()==manifest['checkpoint_sha256'],
                  conditions='Same saved source, CPU single thread, same checkpoint and replay, 10 warmup updates then timed updates; each mode twice in reverse order. No opponent, campaign writes or publishing.')
    reserved()
    (WORK/args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'}), flush=True)


if __name__ == '__main__':
    main()
