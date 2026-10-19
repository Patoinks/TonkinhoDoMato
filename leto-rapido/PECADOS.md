# The Seven Deadly Sins (2026-10-19)

`pecados.py`: seven personalities on G4 Leto (the fast champion), each with its own module, openings, style and
engine settings, at seven difficulty levels. They hook into claudio_agent where the Pantheon did (the opening,
then the style among near-equal safe moves) and into Leto's engine options. Play them with
`vs_stockfish.py --rapido --deus <sin>` or `ab_stockfish.py <name>:pecado=<sin>`.

| Level | Sin | Module | Openings | Score vs SF 2250 (2 ms) | Proof metric (sin vs Leto) |
| ---: | --- | --- | --- | ---: | --- |
| 1 Beginner | **Sloth** | autopilot: London / King's Indian setup from a piece-placement plan; shallow search | the plan (no book) | 0.12 | 3.5 vs 7.2 ms/move ✓ |
| 2 Easy | **Lust** | queen and king hunt: early queen, pieces drawn to the enemy king | Wayward Queen, King's Gambit, Scandinavian | 0.27 | early queen moves 20% vs 11% ✓ |
| 3 Easy+ | **Gluttony** | material appetite: takes any capture within 0.6 of Leto's best losing at most a pawn | gambits accepted, poisoned pawns | 0.31 | captures when possible 33% vs 29% ✓ |
| 4 Intermediate | **Wrath** | king attack: attack units on the king zone, pawn storms, deeper mate search | Evans, Morra, Fried Liver, Mar del Plata | 0.39 | attack units 3.05 vs 2.61 ✓ |
| 5 Advanced | **Envy** | prophylaxis: keeps the opponent's next capture small, restricts it, trades its busiest piece | symmetrical lines | 0.44 | opponent's threat 0.63 vs 0.70 ✓ |
| 6 Advanced+ | **Greed** | hoard and convert: never hangs material, never sacrifices, trades pieces when ahead | exchange lines, Caro-Kann, QGD | 0.46 | own hanging material 0.79 vs 0.91 ✓ |
| 7 Expert | **Pride** | contempt: draws worth little when it thinks it is winning, avoids repetitions and drawish endgames | the most ambitious main lines | 0.55 (Leto 0.56) | draws 21% vs 24% ✓ |

Measured with `ab_stockfish.py` (300 games each, interleaved with neutral Leto) and `prova_pecados.py`.
Sloth's early stop in the C++ iterative deepening is not done: it thinks less through depth 5 and fewer nodes.

## The four rebuilt deities (kept, `personalidades.py`, `repertorios_deuses.py`)

Repertoires written by hand (95-217 lines each, `validar_repertorios.py`), style weights for the target profiles.
Openings only vs openings + style, 300 games each vs SF 2250 (Leto 0.545): Nezha 0.533 / 0.480, Loki 0.487 / 0.430,
Hades 0.490 / 0.410, Anubis 0.465 / 0.385. Book moves per game ~1 -> 2.4-3.1.

## Recalibrated to 0.20 -> 0.70 (2026-10-19)

Final run (`ab_stockfish.py`, 300 games each, one pool; neutral Leto 0.512 in this run, ~0.54 usually):

| Level | Sin | Score | Engine settings that set the level | ms/move | Proof |
| ---: | --- | ---: | --- | ---: | --- |
| 1 Beginner | Sloth | 0.215 | autopilot 16 half-moves, window 0.06 | 7.1 | attack on the king 1.52 vs 2.53 ✓ |
| 2 Easy | Lust | 0.290 | tactic depth 5, window 0.10 | 6.0 | early queen 21% vs 10% ✓ |
| 3 Easy+ | Gluttony | 0.365 | appetite 0.6 / a pawn, safety 2.5 | 7.3 | captures 31% vs 28% ✓ |
| 4 Intermediate | Wrath | 0.412 | window 0.07, mate attack 3 | 8.2 | attack units 3.11 vs 2.53 ✓ |
| 5 Advanced | Envy | 0.455 | tactic depth 7, window 0.012 | 13.7 | opponent's threat 0.62 vs 0.64 ✓ |
| 6 Advanced+ | Greed | 0.552 | tactic depth 7 + check extension, window 0.008 | 19.8 | hanging material 0.78 vs 0.82 ✓ |
| 7 Expert | Pride | 0.623 | tactic depth 8 + check extension, window 0.006 | 44.8 | draws 25% vs 29% ✓ |

0.70 is out of reach: depth 7 + check extension gives ~+0.14 over Leto, depth 8 ~+0.16, depth 9 nothing more
(0.56-0.62 with 3 M nodes) - the ceiling of this network and search against Stockfish 2250 at 2 ms is ~0.63-0.65.
