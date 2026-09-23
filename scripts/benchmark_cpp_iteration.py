"""Compare saved source baseline and current native code in alternating fresh processes."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    import numpy as np
    import torch
    result = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            result.update(b'tensor'); visit(str(v.dtype)); visit(tuple(v.shape))
            result.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            result.update(b'array'); visit(str(v.dtype)); visit(tuple(v.shape)); result.update(v.tobytes())
        elif isinstance(v, dict):
            result.update(b'dict'); visit(len(v))
            for k in sorted(v, key=repr):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list, set)):
            result.update(type(v).__name__.encode()); visit(len(v))
            for item in sorted(v, key=repr) if isinstance(v, set) else v:
                visit(item)
        elif dataclasses.is_dataclass(v):
            result.update(type(v).__name__.encode())
            for field in dataclasses.fields(v):
                visit(field.name); visit(getattr(v, field.name))
        else:
            result.update(pickle.dumps(v, protocol=5))
    visit(value)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--child', choices=['before', 'after'])
    args = parser.parse_args()
    if args.games < 1 or args.output.resolve() == args.checkpoint.resolve():
        parser.error('Use a positive game count and an output different from the checkpoint')
    if args.child:
        source = args.baseline.resolve() if args.child == 'before' else ROOT
        sys.path.append(str(ROOT))  # Compiled extensions stay in the project root.
        sys.path.insert(0, str(source))
        # Let the baseline's own imports resolve against its saved source tree.
        module_path = source/'benchmark_cpp.py' if args.child == 'before' else ROOT/'scripts/benchmark_cpp.py'
        spec = importlib.util.spec_from_file_location('benchmark', module_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.torch.set_num_threads(1)
        blob = args.checkpoint.read_bytes()
        module.trial(blob, True, 1)
        report, evidence = module.trial(blob, True, args.games)
        report.update(version=args.child, evidence_sha256=digest(evidence),
                      checkpoint_sha256=hashlib.sha256(blob).hexdigest())
        print(json.dumps(report), flush=True)
        return
    reports=[]
    for version in ('before', 'after', 'after', 'before'):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 '--baseline', str(args.baseline.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                                 '--games', str(args.games), '--output', str(args.output.resolve()), '--child', version],
                                text=True, capture_output=True, cwd=ROOT)
        if result.returncode:
            raise RuntimeError(result.stderr)
        report=json.loads(result.stdout)
        reports.append(report)
        print(json.dumps(report), flush=True)
        if len({r['evidence_sha256'] for r in reports}) != 1:
            raise AssertionError('Learning or trajectory state differs from the native baseline')
    average=lambda version: sum(r['games_per_second'] for r in reports if r['version']==version)/2
    out=dict(runs=reports, exact_state_match=True, native_speedup=average('after')/average('before'),
             checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==reports[0]['checkpoint_sha256'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='runs'}))


if __name__=='__main__':
    main()
