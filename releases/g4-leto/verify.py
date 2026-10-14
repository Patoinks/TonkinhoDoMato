"""Verify release bytes; optionally prove the original checkpoint/export parity."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
provenance = json.loads((root / 'provenance.json').read_text())
for relative, expected in provenance['files'].items():
    path = root / relative
    assert path.resolve().is_relative_to(root), relative
    content = path.read_bytes()
    assert len(content) == expected['bytes'], relative
    assert hashlib.sha256(content).hexdigest() == expected['sha256'], relative
public = root / 'web/public/tonkai'
manifest = json.loads((public / 'leto.json').read_text())
weights = (public / manifest['binary']).read_bytes()
assert hashlib.sha256(weights).hexdigest() == manifest['sha256'] == provenance['weights_sha256']
for key, attribute in [('native', 'loader'), ('native', 'wasm'), ('pawn_proofs', 'file'), ('opening_repertoire', 'file')]:
    descriptor = manifest[key]
    content = (public / descriptor[attribute]).read_bytes()
    expected = descriptor[attribute + '_sha256'] if key == 'native' else descriptor['sha256']
    assert hashlib.sha256(content).hexdigest() == expected, descriptor[attribute]
if args.checkpoint:
    import torch
    torch.set_num_threads(1)
    checkpoint = torch.load(root / 'models/g4t2-leto-v3.pt', map_location='cpu', weights_only=True)
    exported = b''.join(t.detach().cpu().float().contiguous().numpy().astype('<f4').tobytes()
                        for group in (checkpoint['net'], checkpoint['king']['state_dict']) for t in group.values())
    assert exported == weights, 'Checkpoint tensors differ from web weights.'
print(json.dumps({'files_verified': len(provenance['files']), 'weights_sha256': manifest['sha256'],
                  'checkpoint_tensors_equal': True if args.checkpoint else None}))
