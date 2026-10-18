# Torneio de gerações vs Stockfish 2250 (2 ms), partida do t0467

## Geração 1

128 mutantes, 24 passaram a prova de tempo; 6.5 min.

| finalista | bo400 | ±95% | tempo vs t0467 | mutações (sobre o t0467) |
| --- | ---: | ---: | ---: | --- |
| g0m000 | 0.706 | 0.045 | 0.91 | ahead_draws=True check_extension=1 pawn_convert=0.7842 |
| g0m001 | 0.700 | 0.045 | 0.75 | tactic_nodes=484154 check_extension=1 |
| g0m018 | 0.681 | 0.046 | 0.91 | simplify_margin=0.2036 pawn_convert=0.8641 check_extension=1 |
| g0m007 | 0.661 | 0.046 | 0.72 | check_extension=1 opening_weight=0.04225 exchange_depth=7 simplify_margin=0.02569 |
| campeão (t0467) | 0.629 | | 1.00 | (t0467) |

Campeão mantém-se.

Tendências agora — boas: check_extension↑ +0.104, pawn_weight↑ +0.059, pawn_convert↑ +0.056, opening_weight↓ +0.053, simplify_margin↓ +0.051, pawn_weight↓ +0.048; más: tactic_depth↓ -0.127, piece_values↑ -0.107, scan_keep↓ -0.096, piece_values↓ -0.069, king_weight↑ -0.052, simplify_weight↑ -0.048.

Nota: a prova de tempo da geração 1 usou um só jogo de posições, onde a extensão de xeque quase não dispara;
medidos em três jogos, g0m000 e g0m001 são 1,24x mais lentos que o t0467 (não 0,91 e 0,75). Desde a geração 2 a
prova usa três jogos (~450 posições) e a coroação outros três.

## Geração 2

128 mutantes, 8 passaram a prova de tempo; 7.8 min.

| finalista | bo400 | ±95% | tempo vs t0467 | mutações (sobre o t0467) |
| --- | ---: | ---: | ---: | --- |
| g1m080 | 0.672 | 0.046 | 0.95 | tactic_nodes=484154 check_extension=0 pawn_convert=1 pawn_weight=1.765 opening_weight=0.0532 |
| g1m007 | 0.671 | 0.046 | 0.99 | tactic_nodes=484154 check_extension=0 simplify_margin=0.3654 opening_weight=0.2981 |
| g1m085 | 0.631 | 0.047 | 0.95 | pawn_weight=0.4721 simplify_margin=0.2706 |
| g1m069 | 0.560 | 0.049 | 0.99 | simplify_margin=0.2036 pawn_convert=0.8641 check_extension=1 opening_weight=0.02381 tactic_depth=5 scan_keep=24 |
| campeão (t0467) | 0.647 | | 1.00 | (t0467) |

Campeão mantém-se.

Tendências agora — boas: check_extension↑ +0.097, pawn_weight↑ +0.068, pawn_weight↓ +0.067, pawn_convert↑ +0.057, opening_weight↓ +0.056, simplify_margin↓ +0.047; más: tactic_depth↓ -0.133, piece_values↑ -0.107, scan_keep↓ -0.097, piece_values↓ -0.069, king_weight↑ -0.052, scan_depth↓ -0.047.

## Mapa de tempo (cada mutação sozinha, 3 jogos de posições)

tactic_depth↑ 1.34x, scan_depth↑ 1.26x, check_extension↑ 1.26x, tactic_nodes↑ 1.17x, reply_top↑ 1.14x, pawn_weight↑ 1.13x, opening_weight↑ 1.10x, simplify_margin↓ 1.08x, simplify_margin↑ 1.06x, tactic_rounds↓ 1.06x, piece_values↓ 1.06x, pawn_weight↓ 1.05x, mate_defence↑ 1.04x, tactic_top↑ 1.03x, opening_weight↓ 1.02x, quiesce_promotions↓ 1.01x, mate_defence↓ 1.00x, scan_keep↑ 1.00x, simplify_weight↓ 1.00x, repeat_as_draw↓ 1.00x, mate_attack↑ 1.00x, fifty_horizon↑ 0.99x, value_scaling↓ 0.99x, mate_attack↓ 0.98x, scan_keep↓ 0.98x, piece_values↑ 0.98x, draw_value↑ 0.98x, reply_weight↓ 0.98x, pawn_convert↑ 0.97x, ahead_value↓ 0.97x, draw_value↓ 0.96x, tactic_nodes↓ 0.96x, safety_weight↓ 0.96x, ahead_draws↑ 0.96x, exchange_depth↓ 0.96x, ahead_material↑ 0.95x, pawn_convert↓ 0.95x, exchange_depth↑ 0.94x, king_weight↑ 0.94x, tactic_top↓ 0.94x, reply_weight↑ 0.94x, ahead_material↓ 0.94x, king_weight↓ 0.93x, simplify_weight↑ 0.93x, safety_weight↑ 0.93x, ahead_value↑ 0.92x, fifty_horizon↓ 0.90x, scan_depth↓ 0.88x, tactic_rounds↑ 0.86x, tactic_depth↓ 0.79x

## Geração 3

128 mutantes, 81 passaram a prova de tempo; 8.7 min.

| finalista | bo400 | ±95% | tempo vs t0467 | mutações (sobre o t0467) |
| --- | ---: | ---: | ---: | --- |
| g2m077 | 0.691 | 0.045 | 0.85 | ahead_material=0.2605 pawn_weight=0.1891 |
| g2m053 | 0.642 | 0.047 | 0.91 | exchange_depth=7 safety_weight=6 |
| g2m061 | 0.639 | 0.047 | 0.91 | pawn_weight=0.4721 simplify_margin=0.1207 tactic_rounds=4 |
| g2m040 | 0.621 | 0.048 | 0.84 | tactic_nodes=484154 check_extension=0 pawn_convert=1 pawn_weight=1.765 opening_weight=0.0532 exchange_depth=7 ahead_material=0.566 |
| campeão (antigo) | 0.656 | | 1.00 | (t0467) |

**Novo campeão: g2m077** (+0.035).

Tendências agora — boas: check_extension↑ +0.094, reply_top↑ +0.075, ahead_material↑ +0.056, pawn_weight↑ +0.043, pawn_weight↓ +0.037, exchange_depth↑ +0.036; más: tactic_depth↓ -0.147, scan_keep↓ -0.099, piece_values↑ -0.068, piece_values↓ -0.061, king_weight↑ -0.056, scan_depth↓ -0.040.

## Geração 4

128 mutantes, 73 passaram a prova de tempo; 5.4 min.

| finalista | bo400 | ±95% | tempo vs t0467 | mutações (sobre o t0467) |
| --- | ---: | ---: | ---: | --- |
| g3m031 | 0.630 | 0.047 | 0.90 | tactic_nodes=484154 check_extension=1 pawn_convert=1 pawn_weight=1.765 opening_weight=0.2953 exchange_depth=7 ahead_material=0.566 phase_values=1.000,2.557,3.273,4.778,10.874;1.000,2.557,3.273,4.778,10.874;1.000,2.557,3.273,4.778,10.874 scan_depth=2 |
| g3m076 | 0.620 | 0.048 | 0.98 | pawn_weight=0.4721 simplify_margin=0.1207 tactic_rounds=4 king_weight=0.1988 safety_weight=6 simplify_weight=0.01059 exchange_depth=7 mate_attack=1 |
| g3m125 | 0.611 | 0.048 | 0.94 | tactic_nodes=484154 check_extension=0 pawn_convert=1 pawn_weight=1.765 opening_weight=0.0532 exchange_depth=7 ahead_material=0.8104 fifty_horizon=29 |
| g3m062 | 0.610 | 0.048 | 0.93 | tactic_nodes=484154 check_extension=0 pawn_convert=1 pawn_weight=1.765 opening_weight=0.0532 exchange_depth=7 ahead_material=0.6126 draw_value=0.3532 safety_weight=4.215 |
| campeão (g2m077) | 0.669 | | 1.00 | ahead_material=0.2605 pawn_weight=0.1891 |

Campeão mantém-se.

Tendências agora — boas: check_extension↑ +0.095, pawn_weight↓ +0.056, ahead_material↑ +0.056, reply_top↑ +0.054, pawn_weight↑ +0.042, tactic_top↑ +0.041; más: tactic_depth↓ -0.149, scan_keep↓ -0.096, king_weight↑ -0.049, exchange_depth↓ -0.044, piece_values↓ -0.044, piece_values↑ -0.043.
