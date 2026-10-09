"""Own-experience collection and organic compact-network/policy training.

Stockfish is only an opponent. Labels come from the frozen checkpoint 600,
its exact rule-based mate planner and the actual outcomes of its own games.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths
from v2.agent import load_v2, OwnPolicy
from v2.rules import adjudicate, draw_config

_worker = {}


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def init_collection(checkpoint, engine, out):
    import torch
    import psutil_free_priority
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from tonkai.uci_opponent import UCIOpponent
    _worker.update(checkpoint=checkpoint, engine=UCIOpponent(engine, 1800, 0.002), out=Path(out))


def collect_game(game, seed):
    import chess
    import numpy as np
    teacher = load_v2(_worker['checkpoint'], variant='finish', rng=np.random.default_rng([seed, game, 1]))
    me = game % 2 == 0
    board = chess.Board()
    engine = _worker['engine']
    engine.reset()
    greedy = teacher.cfg
    explore = replace(greedy, temperature=0.005)
    start = time.perf_counter()
    while adjudicate(board) is None and board.ply() < 300:
        if board.turn == me:
            teacher.cfg = explore if board.ply() < 16 else greedy
            move = teacher.choose(board)
        else:
            move = engine.choose(board, np.random.default_rng([seed, game]))
        if move not in board.legal_moves:
            raise AssertionError('Illegal collection move')
        board.push(move)
    outcome = adjudicate(board)
    # Consume the actor's actual cached trajectory for learning preparation.
    import torch
    from v2.inference import cached_surprises
    from claudio_agent import native_board
    from tonkai import fast_rules
    _, surprises = cached_surprises(teacher, board)
    prefix = min(29, len(teacher.real_memory))
    with torch.inference_mode():
        context = teacher.net.context_from_latent(torch.cat(teacher.real_latent[:prefix]),
                                                 torch.cat(teacher.real_memory[:prefix]))
        own_view = teacher.net.survival(context)[0].mean(-1).numpy()
        fast = native_board(board.root())
        other_rows = [np.asarray(fast.encode_here(not me), dtype=np.float32)]
        for move in board.move_stack[:prefix-1]:
            fast.push(fast_rules.code(move))
            other_rows.append(np.asarray(fast.encode_here(not me), dtype=np.float32))
        latent = teacher.net.encode(torch.from_numpy(np.stack(other_rows)))
        memory, contexts = teacher.net.initial_memory(1), []
        for i in range(len(latent)):
            memory = teacher.net.memory_cell(latent[i:i+1], memory)
            contexts.append(teacher.net.context_from_latent(latent[i:i+1], memory))
        other_view = teacher.net.survival(torch.cat(contexts))[0].mean(-1).numpy()
    row = {'game': game, 'teacher_white': me, 'moves': ' '.join(m.uci() for m in board.move_stack),
           'winner': None if outcome is None or outcome['winner'] is None else ('white' if outcome['winner'] else 'black'),
           'end': outcome['end'] if outcome else 'max_plies', 'plies': board.ply(),
           'seconds': time.perf_counter() - start, 'stats': teacher.stats,
           'own_opening_view': own_view.tolist(), 'other_opening_view': other_view.tolist(),
           'cached_trajectory_positions': len(teacher.real_rows), 'surprise_mean': float(surprises.mean())}
    write(_worker['out'] / 'experience-games' / f'{game:04d}.json', row)
    return row


def collect_batch(games, seed):
    try:
        return [collect_game(game, seed) for game in games]
    finally:
        # UCI owns an event-loop thread; close it before the worker exits.
        _worker['engine'].close()


def collect(args):
    out = Path(args.out)
    (out / 'experience-games').mkdir(parents=True, exist_ok=True)
    existing = {int(file.stem) for file in (out / 'experience-games').glob('*.json')}
    todo = [g for g in range(args.games) if g not in existing]
    begun = time.perf_counter()
    rows = [json.loads(file.read_text(encoding='utf-8')) for file in sorted((out / 'experience-games').glob('*.json'))]
    with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn'), initializer=init_collection,
                             initargs=(args.checkpoint, args.engine, str(out))) as pool:
        batches = [todo[i::args.workers] for i in range(min(args.workers, len(todo)))]
        futures = [pool.submit(collect_batch, batch, args.seed) for batch in batches]
        for future in as_completed(futures):
            batch_rows = future.result()
            rows.extend(batch_rows)
            write(out / 'preparation-status.json', {'phase': 'own experience', 'games': len(rows), 'target': args.games,
                                                   'seconds': time.perf_counter()-begun})
            print(f"own-experience {len(rows)}/{args.games}", flush=True)
    rows.sort(key=lambda r: r['game'])
    write(out / 'experience.json', {'checkpoint': args.checkpoint, 'teacher_only_moves': True, 'games': rows})


def make_book(args):
    import torch
    torch.set_num_threads(1)
    import experiencia
    import chess
    from collections import defaultdict
    out = Path(args.out)
    games = json.loads((out / 'experience.json').read_text(encoding='utf-8'))['games']
    # Both views belong to the same frozen teacher; no engine evaluations are used.
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in games:
        mine, theirs = row['own_opening_view'], row['other_opening_view']
        board = chess.Board()
        moves = row['moves'].split()
        for ply, uci in enumerate(moves[:16]):
            if board.turn == row['teacher_white']:
                side = 'white' if board.turn else 'black'
                then = min(ply + 12, len(mine) - 1)
                good = row['winner'] == side and mine[then] >= theirs[then]
                cell = cells[experiencia.key_of(board)][uci]
                cell[0] += 1
                cell[1] += int(good)
            board.push_uci(uci)
    book = {'format': 'claudio-own-book-v1', 'source_checkpoint': args.checkpoint,
            'games': len(games), 'plies': 16, 'later': 12,
            'rule': 'Own move; won; own network side at least as good as opposite view 12 plies later',
            'engine_moves_as_targets': False, 'positions': dict(cells)}
    write(out / 'opening-book.json', book)
    print(f'own book: {len(cells)} positions from {len(games)} actual own games', flush=True)


def make_dataset(args):
    import chess
    import numpy as np
    import torch
    import psutil_free_priority
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from tonkai import fast_rules
    from claudio_agent import native_board
    out = Path(args.out)
    archived = [json.loads(line) for line in (paths.ROOT / 'claudio/results/matches/g16-vs-g15-100-bo30.jsonl').read_text().splitlines() if line.strip()]
    archived.sort(key=lambda r: r['game'])
    experience = json.loads((out / 'experience.json').read_text(encoding='utf-8'))['games'] if (out / 'experience.json').exists() else []
    games = archived + experience
    eligible = [(g, ply) for g, row in enumerate(games) for ply in range(len(row['moves'].split()))
                if (g < len(archived) and ply % 4 == 0) or (g >= len(archived) and ply < 24)]
    count = min(args.positions, len(eligible))
    indexes = sorted({round(i*(len(eligible)-1)/(count-1)) for i in range(count)})
    selected = {eligible[i] for i in indexes}
    chunks = {key: [] for key in ('x', 'memory', 'latent', 'context', 'after', 'heads', 'predict', 'order_base', 'order_features')}
    groups = []
    offset = 0
    with torch.inference_mode():
        for game_id, row in enumerate(games):
            players = {side: load_v2(args.checkpoint, variant='finish', rng=np.random.default_rng([8901, game_id, int(side)]))
                       for side in (False, True)}
            for player in players.values(): player.options.capture_policy_inputs=True
            board = chess.Board()
            for ply, uci in enumerate(row['moves'].split()):
                player = players[board.turn]
                if (game_id, ply) in selected and adjudicate(board) is None:
                    player._catch_up(board)
                    memory = player.memory.detach().clone()
                    choice = player.choose(board,terminal_checked=True)
                    fast = native_board(board)
                    codes = list(fast.legal_moves())
                    x = torch.from_numpy(fast_rules.encode_after_moves(fast, codes, board.turn))
                    latent = player.net.encode(x)
                    contexts, after = player.net.candidates(x, memory)
                    heads, _ = player.net.survival(contexts)
                    logits = player.net.predict_logits(contexts)
                    chunks['x'].append(x.numpy())
                    chunks['memory'].append(memory.expand(len(codes), -1).numpy())
                    chunks['latent'].append(latent.numpy())
                    chunks['context'].append(contexts.numpy())
                    chunks['after'].append(after.numpy())
                    chunks['heads'].append(heads.numpy())
                    chunks['predict'].append(logits.numpy())
                    snapshot=player.order_snapshot
                    top=snapshot['top'] if snapshot else []
                    chosen_index=codes.index(fast_rules.code(choice))
                    base=np.asarray(snapshot['scores']-max(snapshot['scores'][top]),dtype=np.float32) if top else np.zeros(len(codes),dtype=np.float32)
                    features=(np.column_stack([np.tanh(snapshot['gains']/3),np.tanh(snapshot['nets']/3),base]).astype(np.float32)
                              if top else np.zeros((len(codes),3),dtype=np.float32))
                    chunks['order_base'].append(base)
                    chunks['order_features'].append(features)
                    groups.append({'start': offset, 'size': len(codes), 'target': codes.index(fast_rules.code(choice)),
                                   'game': game_id, 'ply': ply, 'proof': player.stats['planner_hits'] > 0,
                                   'top':top,'policy_target':top.index(chosen_index) if chosen_index in top else None,
                                   'moves': [chess.Move(c & 63, (c >> 6) & 63, (c >> 12) or None).uci() for c in codes]})
                    offset += len(codes)
                    if len(groups) % 50 == 0:
                        write(out / 'preparation-status.json', {'phase': 'own search labels', 'positions': len(groups), 'target': count})
                        print(f'own labels {len(groups)}/{count}; {offset} candidate rows', flush=True)
                else:
                    player._catch_up(board)
                board.push_uci(uci)
    arrays = {key: np.concatenate(value).astype(np.float32) for key, value in chunks.items()}
    np.savez_compressed(out / 'own-dataset.npz', **arrays)
    write(out / 'own-dataset.json', {'groups': groups, 'rows': offset,
                                   'teacher_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                                   'teacher': 'checkpoint600 + own exact mate proof; no engine scores/moves as targets'})
    print(f'own dataset: {len(groups)} roots / {offset} candidates', flush=True)


def compact_initial(teacher, architecture):
    """Merge nearly duplicated Net2Net encoder units using only the teacher's weights."""
    import torch
    from claudio_net import build_net
    shape = dict(architecture, hidden_size=256, residual_hidden=256)
    student = build_net(shape)
    src, dst = teacher.state_dict(), student.state_dict()
    compatible = {k: value for k, value in src.items() if k in dst and dst[k].shape == value.shape}
    student.load_state_dict(compatible, strict=False)
    first, second = teacher.encoder[0], teacher.encoder[2]
    with torch.no_grad():
        # First 256 original units; the extra Net2Net copies are assigned to their nearest unit.
        base = first.weight[:256]
        assignment = torch.cat([torch.arange(256), torch.cdist(first.weight[256:], base).argmin(1)])
        for unit in range(256):
            members = torch.nonzero(assignment == unit).flatten()
            contribution = second.weight[:, members].abs().mean(0).clamp_min(1e-8)
            shares = contribution / contribution.sum()
            student.encoder[0].weight[unit].copy_((first.weight[members] * shares[:, None]).sum(0))
            student.encoder[0].bias[unit].copy_((first.bias[members] * shares).sum())
            student.encoder[2].weight[:, unit].copy_(second.weight[:, members].sum(1))
        student.encoder[2].bias.copy_(second.bias)
        if architecture.get('residuals'):
            block = teacher.encoder[-1]
            selected = block.fc2.weight.abs().mean(0).argsort(descending=True)[:256]
            student.encoder[-1].fc1.weight.copy_(block.fc1.weight[selected])
            student.encoder[-1].fc1.bias.copy_(block.fc1.bias[selected])
            student.encoder[-1].fc2.weight.copy_(block.fc2.weight[:, selected])
            student.encoder[-1].fc2.bias.copy_(block.fc2.bias)
    return student, shape


def learn(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    import psutil_free_priority
    from claudio_load import load_claudio, save_claudio
    from claudio_agent import ClaudioConfig
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(9021)
    out = Path(args.out)
    teacher, values, config, history, king = load_claudio(args.checkpoint, with_config=True, with_king=True)
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.reuse_compact:
        student = load_claudio(out / 'compact.pt')[0]
        shape = torch.load(out / 'compact.pt', map_location='cpu', weights_only=False)['architecture']
    else:
        student, shape = compact_initial(teacher, original['architecture'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    student.to(device).eval().requires_grad_(False)
    student.encoder.requires_grad_(True)
    policy = OwnPolicy(student.latent_size).to(device)
    arrays = np.load(out / 'own-dataset.npz')
    groups = json.loads((out / 'own-dataset.json').read_text(encoding='utf-8'))['groups']
    data = {key: torch.from_numpy(arrays[key]).to(device) for key in arrays.files}
    training_groups = [g for g in groups if g['game'] % 5 != 0]
    validation_groups = [g for g in groups if g['game'] % 5 == 0]
    policy_groups=[g for g in training_groups if g['policy_target'] is not None and len(g['top'])>1]
    policy_validation=[g for g in validation_groups if g['policy_target'] is not None and len(g['top'])>1]
    spreads=[float(np.std(arrays['order_base'][g['start']+np.asarray(g['top'])])) for g in policy_groups]
    temperature=max(config['critical_gap'],min(config['value_spread_ref'],float(np.median(spreads))))
    policy.temperature.fill_(temperature)
    train_rows = torch.tensor([i for g in training_groups for i in range(g['start'], g['start']+g['size'])], device=device)
    valid_rows = torch.tensor([i for g in validation_groups for i in range(g['start'], g['start']+g['size'])], device=device)
    optimizer = torch.optim.AdamW(student.encoder.parameters(), lr=2e-4, weight_decay=1e-5)
    best = json.loads((out / 'learning.json').read_text())['best_value_mae'] if args.reuse_compact else float('inf')
    best_state = {k:v.detach().cpu().clone() for k,v in student.state_dict().items()} if args.reuse_compact else None
    history_loss = []
    for step in range(args.steps):
        started = time.perf_counter()
        ids = train_rows[torch.randint(len(train_rows), (256,), device=device)]
        latent = student.encode(data['x'][ids])
        after = student.memory_cell(latent, data['memory'][ids])
        contexts = student.context_from_latent(latent, after)
        heads, _ = student.survival(contexts)
        loss = (F.mse_loss(latent, data['latent'][ids]) + F.mse_loss(contexts, data['context'][ids])
                + F.mse_loss(after, data['after'][ids]) + 4*F.mse_loss(heads, data['heads'][ids]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.encoder.parameters(), 1.0)
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
            time.sleep(max(0.05, time.perf_counter()-started))
        if step % 100 == 0 or step == args.steps-1:
            with torch.inference_mode():
                sample = valid_rows[:4096]
                context, _ = student.candidates(data['x'][sample], data['memory'][sample])
                h, _ = student.survival(context)
                value_error = float((h-data['heads'][sample]).abs().mean())
                latent_error = float(F.mse_loss(student.encode(data['x'][sample]), data['latent'][sample]))
            if value_error < best:
                best = value_error
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            row = {'step': step+1, 'value_mae': value_error, 'latent_mse': latent_error, 'train_loss': float(loss.detach())}
            history_loss.append(row)
            write(out / 'preparation-status.json', {'phase': 'organic compact encoder', 'step': step+1, 'target': args.steps, **row})
            print('compact '+json.dumps(row), flush=True)
    student.load_state_dict(best_state)
    student.eval().requires_grad_(False)
    # Learn a separate own-choice policy; existing opponent-prediction head is preserved.
    opt_policy = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=1e-5)
    rng = np.random.default_rng(9107)
    for step in range(args.policy_steps):
        started = time.perf_counter()
        chosen = [policy_groups[int(rng.integers(len(policy_groups)))] for _ in range(24)]
        ids = torch.tensor([group['start']+i for group in chosen for i in group['top']], device=device)
        with torch.no_grad():
            contexts, _ = student.candidates(data['x'][ids], data['memory'][ids])
        features=data['order_features'][ids].clone()
        features[:,2]=(features[:,2]/temperature).clamp(-10,0)
        pieces = policy(contexts,features,data['order_base'][ids]).split([len(group['top']) for group in chosen])
        losses = []
        for group, logits in zip(chosen, pieces, strict=True):
            losses.append(F.cross_entropy(logits[None], torch.tensor([group['policy_target']], device=device)))
        loss = torch.stack(losses).mean()
        opt_policy.zero_grad(set_to_none=True)
        loss.backward()
        opt_policy.step()
        if device == 'cuda':
            torch.cuda.synchronize()
            time.sleep(max(0.05, time.perf_counter()-started))
        if step % 100 == 0 or step == args.policy_steps-1:
            print(f'own policy {step+1}/{args.policy_steps}: loss {float(loss.detach()):.5f}', flush=True)
            write(out / 'preparation-status.json', {'phase': 'own policy', 'step': step+1, 'target': args.policy_steps})
    correct,base_correct = 0,0
    calibration=[]
    with torch.inference_mode():
        for group in policy_validation:
            ids=torch.tensor([group['start']+i for i in group['top']],device=device)
            context, _ = student.candidates(data['x'][ids], data['memory'][ids])
            features=data['order_features'][ids].clone()
            features[:,2]=(features[:,2]/temperature).clamp(-10,0)
            ranks=policy(context,features,data['order_base'][ids])
            good=int(ranks.argmax())==group['policy_target']
            base_good=int(data['order_base'][ids].argmax())==group['policy_target']
            correct+=int(good);base_correct+=int(base_good)
            calibration.append((float(ranks.softmax(0).max()),good,base_good))
    threshold=1.01
    for level in np.arange(0.1,0.91,0.05):
        selected=[row for row in calibration if row[0]>=level]
        if len(selected)>=20 and sum(row[1] for row in selected)/len(selected)>=0.9 and sum(row[1] for row in selected)>=sum(row[2] for row in selected):
            threshold=float(level);break
    student.cpu()
    save_claudio(out / 'compact.pt', student, shape, values, ClaudioConfig.with_settings([], base=draw_config(config)),
                 history+[{'g16_v2_organic_distillation': True, 'source_games': 600, 'best_value_mae': best}], king_model=king)
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, out / 'policy.pt')
    report = {'teacher_parameters': teacher.n_parameters(), 'compact_parameters': student.n_parameters(),
              'best_value_mae': best, 'policy_validation_top1': correct/max(1,len(policy_validation)),
              'policy_original_order_accuracy':base_correct/max(1,len(policy_validation)),
              'policy_confidence_threshold':threshold,'policy_temperature':temperature,
              'policy_validation_roots': len(policy_validation), 'source_games': 600,
              'teacher_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'device': device, 'engine_scores_as_targets': False, 'loss_history': history_loss}
    write(out / 'learning.json', report)
    print('organic learning complete '+json.dumps({k:v for k,v in report.items() if k!='loss_history'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect','book','dataset','learn'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--engine', default=str(paths.STOCKFISH))
    parser.add_argument('--games', type=int, default=80)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=6301)
    parser.add_argument('--positions', type=int, default=900)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--policy-steps', type=int, default=600)
    parser.add_argument('--reuse-compact', action='store_true')
    args = parser.parse_args()
    {'collect':collect,'book':make_book,'dataset':make_dataset,'learn':learn}[args.stage](args)
