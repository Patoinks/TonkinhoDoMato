# Torneio de mutações com prova de tempo vs Stockfish 2250 (2 ms)

## Prova de tempo

470 vivos, 554 mortos (mais lentos que a base). Tempo relativo à base: mínimo 0.23, mediana 1.02, máximo 3.20.

## Fase 1: 470 x bo3

1413 jogos, 0.43 s/jogo

| # | id | score | ±95% | V/E/D | tempo vs base | ms/lance (prova) | mutações |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | t0359 | 1.000 | 0.253 | 3/0/0 | 0.56 | 4.67 | repeat_as_draw=False scan_depth=2 tactic_depth=4 opening_weight=0.3572 check_extension=1 reply_weight=0.1341 reply_top=3 mate_defence=1 |
| 2 | t0885 | 1.000 | 0.253 | 3/0/0 | 0.75 | 6.32 | reply_top=1 ahead_material=0.2886 value_scaling=True pawn_weight=0.143 scan_depth=2 pawn_convert=1 |
| 3 | t0560 | 1.000 | 0.253 | 3/0/0 | 0.87 | 7.09 | tactic_nodes=268553 exchange_depth=7 king_weight=0.3674 ahead_material=0.5233 draw_value=0.2 mate_defence=1 |
| 4 | t0721 | 1.000 | 0.253 | 3/0/0 | 0.98 | 7.78 | repeat_as_draw=False tactic_nodes=1018048 opening_weight=0.2114 pawn_convert=1 ahead_draws=True tactic_top=8 quiesce_promotions=False reply_weight=0.9429 |
| 5 | t0476 | 1.000 | 0.253 | 3/0/0 | 0.98 | 7.96 | simplify_margin=2.859 mate_defence=3 value_scaling=True fifty_horizon=2 reply_top=1 phase_values=1.000,2.231,3.371,5.407,8.165;1.000,2.231,3.371,5.407,8.165;1.000,2.231,3.371,5.407,8.165 exchange_depth=4 ahead_value=0.8899 safety_weight=2.892 mate_attack=1 |
| 6 | t0079 | 1.000 | 0.253 | 3/0/0 | 0.99 | 7.91 | quiesce_promotions=False fifty_horizon=2 draw_value=0.4653 simplify_margin=1.886 simplify_weight=0.1202 |
| 7 | t0271 | 1.000 | 0.253 | 3/0/0 | 0.99 | 8.11 | mate_attack=3 tactic_rounds=1 pawn_convert=0.2082 exchange_depth=6 pawn_weight=0.1924 repeat_as_draw=False mate_defence=1 value_scaling=True scan_keep=8 check_extension=1 |
| 8 | t0605 | 1.000 | 0.253 | 3/0/0 | 1.00 | 8.01 | opening_weight=0.08134 safety_weight=6 tactic_rounds=4 fifty_horizon=4 ahead_draws=True value_scaling=True |
| 9 | t0529 | 0.833 | 0.422 | 2/1/0 | 0.51 | 4.14 | tactic_rounds=4 king_weight=0.4081 tactic_depth=5 simplify_weight=0.02408 scan_depth=2 draw_value=0.6 ahead_material=0.8092 tactic_nodes=172539 scan_keep=14 |
| 10 | t0559 | 0.833 | 0.422 | 2/1/0 | 0.64 | 5.09 | simplify_margin=1.212 opening_weight=0.05147 scan_depth=0 reply_weight=0.848 pawn_weight=0.4754 exchange_depth=7 |
| 11 | t0289 | 0.833 | 0.422 | 2/1/0 | 0.68 | 5.49 | repeat_as_draw=False mate_attack=3 tactic_rounds=2 ahead_material=0.4954 scan_depth=1 opening_weight=0.08165 safety_weight=3.095 tactic_top=12 draw_value=0.6 pawn_convert=0.8554 |
| 12 | t0632 | 0.833 | 0.422 | 2/1/0 | 0.70 | 5.58 | ahead_material=1 pawn_convert=0.3225 opening_weight=0.4348 king_weight=0.1109 fifty_horizon=18 simplify_margin=0.553 scan_depth=1 |
| 13 | t0767 | 0.833 | 0.422 | 2/1/0 | 0.71 | 5.66 | quiesce_promotions=False tactic_top=6 mate_attack=3 fifty_horizon=9 tactic_depth=4 simplify_weight=0.1329 mate_defence=3 safety_weight=3.439 |
| 14 | t0308 | 0.833 | 0.422 | 2/1/0 | 0.73 | 5.89 | scan_depth=2 reply_top=1 tactic_nodes=1500000 reply_weight=0.3608 scan_keep=13 repeat_as_draw=False safety_weight=1.466 |
| 15 | t0794 | 0.833 | 0.422 | 2/1/0 | 0.74 | 5.97 | simplify_margin=1.328 tactic_top=7 ahead_material=1 safety_weight=6 opening_weight=0.02585 simplify_weight=0.3971 mate_attack=1 quiesce_promotions=False scan_depth=2 |
| 16 | t0911 | 0.833 | 0.422 | 2/1/0 | 0.74 | 6.23 | simplify_margin=0.3163 ahead_value=0.7123 value_scaling=True opening_weight=0.02766 scan_depth=2 |
| 17 | t1015 | 0.833 | 0.422 | 2/1/0 | 0.75 | 6.24 | opening_weight=0.06132 repeat_as_draw=False tactic_rounds=1 simplify_margin=0.5171 pawn_weight=1.043 value_scaling=True tactic_depth=5 ahead_material=1 ahead_draws=True |
| 18 | t0890 | 0.833 | 0.422 | 2/1/0 | 0.78 | 6.41 | value_scaling=True mate_defence=3 quiesce_promotions=False scan_depth=1 safety_weight=6 repeat_as_draw=False scan_keep=18 tactic_nodes=1290366 |
| 19 | t0616 | 0.833 | 0.422 | 2/1/0 | 0.79 | 6.38 | value_scaling=True simplify_weight=0.4446 ahead_material=0.3005 tactic_nodes=1500000 opening_weight=0.1671 scan_depth=2 safety_weight=6 |
| 20 | t0968 | 0.833 | 0.422 | 2/1/0 | 0.80 | 6.66 | pawn_weight=0.5628 fifty_horizon=7 exchange_depth=7 check_extension=1 scan_depth=1 simplify_margin=0.6247 ahead_material=0.3829 |
| 21 | t0366 | 0.833 | 0.422 | 2/1/0 | 0.81 | 6.32 | ahead_material=1 reply_weight=0.01926 reply_top=2 tactic_rounds=2 scan_depth=1 |
| 22 | t0008 | 0.833 | 0.422 | 2/1/0 | 0.83 | 8.60 | king_weight=0.4412 tactic_top=4 opening_weight=0.03536 reply_top=1 phase_values=1.000,1.741,2.336,4.061,11.839;1.000,1.741,2.336,4.061,11.839;1.000,1.741,2.336,4.061,11.839 |
| 23 | t0670 | 0.833 | 0.422 | 2/1/0 | 0.87 | 6.97 | tactic_top=6 scan_depth=1 repeat_as_draw=False value_scaling=True reply_weight=0.09228 quiesce_promotions=False fifty_horizon=9 check_extension=1 phase_values=1.000,1.595,1.931,4.039,11.525;1.000,1.595,1.931,4.039,11.525;1.000,1.595,1.931,4.039,11.525 |
| 24 | t0425 | 0.833 | 0.422 | 2/1/0 | 0.87 | 7.51 | pawn_convert=0.2646 quiesce_promotions=False exchange_depth=4 mate_attack=1 scan_keep=8 phase_values=1.000,2.693,2.842,4.478,7.121;1.000,2.693,2.842,4.478,7.121;1.000,2.693,2.842,4.478,7.121 simplify_margin=1.273 ahead_material=0.2935 tactic_top=6 |
| 25 | t0730 | 0.833 | 0.422 | 2/1/0 | 0.87 | 7.02 | exchange_depth=4 tactic_top=7 pawn_weight=0.5369 scan_keep=16 safety_weight=3.208 |
| 26 | t0987 | 0.833 | 0.422 | 2/1/0 | 0.88 | 7.37 | fifty_horizon=1 ahead_material=1 tactic_top=7 draw_value=0.4843 simplify_margin=1.72 exchange_depth=7 pawn_convert=0.9093 reply_weight=0.6055 |
| 27 | t0650 | 0.833 | 0.422 | 2/1/0 | 0.90 | 7.26 | exchange_depth=7 mate_attack=3 simplify_margin=0.1616 safety_weight=3.069 ahead_material=0.5766 pawn_convert=0.6579 mate_defence=1 simplify_weight=0.4843 fifty_horizon=19 |
| 28 | t0517 | 0.833 | 0.422 | 2/1/0 | 0.90 | 7.28 | check_extension=1 opening_weight=0.2377 scan_depth=1 ahead_material=1 value_scaling=True ahead_value=0.5 |
| 29 | t0439 | 0.833 | 0.422 | 2/1/0 | 0.91 | 7.71 | exchange_depth=6 tactic_nodes=1155734 value_scaling=True mate_defence=1 reply_top=1 ahead_draws=True fifty_horizon=15 |
| 30 | t0691 | 0.833 | 0.422 | 2/1/0 | 0.91 | 7.41 | reply_top=1 draw_value=0.4226 pawn_weight=0.3441 exchange_depth=7 quiesce_promotions=False |
| 31 | t0426 | 0.833 | 0.422 | 2/1/0 | 0.94 | 7.92 | tactic_rounds=1 king_weight=0.3782 pawn_convert=0.0296 tactic_top=4 opening_weight=0.1161 mate_attack=1 tactic_nodes=1500000 mate_defence=3 |
| 32 | t0040 | 0.833 | 0.422 | 2/1/0 | 0.94 | 7.48 | ahead_material=0.1527 opening_weight=0.1511 fifty_horizon=25 tactic_nodes=223248 check_extension=1 reply_top=1 scan_depth=1 |
| 33 | t0116 | 0.833 | 0.422 | 2/1/0 | 0.94 | 7.76 | tactic_depth=5 pawn_weight=0.5818 ahead_value=0.7378 scan_keep=12 reply_top=4 safety_weight=6 |
| 34 | t0757 | 0.833 | 0.422 | 2/1/0 | 0.94 | 7.58 | king_weight=0.109 tactic_nodes=148783 reply_weight=0.9543 ahead_draws=True exchange_depth=4 ahead_value=0.8044 mate_attack=1 pawn_weight=1.759 value_scaling=True fifty_horizon=24 |
| 35 | t0862 | 0.833 | 0.422 | 2/1/0 | 0.94 | 7.66 | ahead_draws=True opening_weight=0.3812 draw_value=0.436 check_extension=1 ahead_material=0.7203 scan_depth=2 fifty_horizon=29 pawn_weight=0.4881 |
| 36 | t0295 | 0.833 | 0.422 | 2/1/0 | 0.95 | 7.69 | value_scaling=True ahead_value=0.5145 exchange_depth=7 check_extension=1 simplify_margin=0.2402 pawn_convert=0.03647 tactic_depth=5 ahead_draws=True |
| 37 | t0396 | 0.833 | 0.422 | 2/1/0 | 0.96 | 8.27 | mate_attack=3 simplify_margin=2.002 check_extension=1 phase_values=1.000,1.762,2.083,4.964,11.600;1.000,1.762,2.083,4.964,11.600;1.000,1.762,2.083,4.964,11.600 scan_depth=2 king_weight=0.4399 pawn_weight=0.594 pawn_convert=0.7447 fifty_horizon=14 |
| 38 | t0534 | 0.833 | 0.422 | 2/1/0 | 0.96 | 7.47 | safety_weight=2.652 mate_attack=3 opening_weight=0.2858 tactic_rounds=1 fifty_horizon=2 exchange_depth=7 ahead_value=0.5255 |
| 39 | t0502 | 0.833 | 0.422 | 2/1/0 | 0.97 | 7.83 | pawn_weight=1.545 simplify_margin=0.1738 simplify_weight=0.4993 repeat_as_draw=False reply_weight=0.9175 draw_value=0.3031 quiesce_promotions=False |
| 40 | t0825 | 0.833 | 0.422 | 2/1/0 | 0.97 | 7.64 | draw_value=0.385 pawn_convert=0.2145 value_scaling=True opening_weight=0.1259 exchange_depth=6 tactic_rounds=4 mate_defence=3 |
| 41 | t0888 | 0.833 | 0.422 | 2/1/0 | 0.97 | 7.91 | safety_weight=6 tactic_top=14 exchange_depth=6 mate_defence=1 reply_weight=0.07506 phase_values=1.000,2.246,2.094,4.676,7.822;1.000,2.246,2.094,4.676,7.822;1.000,2.246,2.094,4.676,7.822 king_weight=0.4968 |
| 42 | t0638 | 0.833 | 0.422 | 2/1/0 | 0.98 | 8.00 | draw_value=0.6 pawn_convert=0.2578 tactic_nodes=147294 fifty_horizon=25 simplify_weight=0.2771 |
| 43 | t0193 | 0.833 | 0.422 | 2/1/0 | 0.99 | 7.93 | scan_keep=7 check_extension=1 draw_value=0.2742 pawn_convert=1 exchange_depth=6 simplify_weight=0.1221 reply_weight=0.1775 tactic_nodes=1500000 |
| 44 | t0379 | 0.833 | 0.422 | 2/1/0 | 0.99 | 8.16 | simplify_weight=0.1228 safety_weight=4.701 mate_defence=3 tactic_top=11 check_extension=1 reply_top=3 repeat_as_draw=False simplify_margin=2.964 tactic_rounds=2 tactic_depth=4 |
| 45 | t0385 | 0.833 | 0.422 | 2/1/0 | 0.99 | 8.11 | tactic_nodes=1500000 ahead_material=0.7913 scan_keep=22 simplify_margin=2.015 opening_weight=0.4987 repeat_as_draw=False |
| 46 | t0348 | 0.833 | 0.422 | 2/1/0 | 0.99 | 7.82 | fifty_horizon=19 ahead_value=0.9 quiesce_promotions=False pawn_convert=1 tactic_nodes=292624 opening_weight=0.1189 tactic_rounds=1 |
| 47 | t0828 | 0.833 | 0.422 | 2/1/0 | 0.99 | 8.00 | mate_defence=1 pawn_weight=0.1945 opening_weight=0.05605 simplify_margin=0.2566 tactic_rounds=1 king_weight=0.4266 quiesce_promotions=False tactic_nodes=279065 reply_weight=0.05763 repeat_as_draw=False |
| 48 | t0591 | 0.833 | 0.422 | 2/1/0 | 0.99 | 8.10 | reply_top=2 reply_weight=0.3301 tactic_rounds=1 fifty_horizon=14 check_extension=1 pawn_weight=0.1856 ahead_value=0.8051 simplify_margin=0.5324 repeat_as_draw=False scan_depth=1 |
| 49 | t0137 | 0.833 | 0.422 | 2/1/0 | 1.00 | 7.99 | tactic_nodes=141387 reply_weight=0.965 pawn_convert=0.3019 mate_attack=1 quiesce_promotions=False |
| 50 | t0499 | 0.667 | 0.533 | 1/2/0 | 0.36 | 2.96 | tactic_depth=5 pawn_weight=1.439 tactic_top=4 ahead_draws=True scan_depth=1 scan_keep=10 repeat_as_draw=False |
| 51 | t0849 | 0.667 | 0.533 | 1/2/0 | 0.58 | 4.67 | reply_weight=0.195 ahead_value=0.5826 tactic_depth=4 repeat_as_draw=False exchange_depth=7 simplify_margin=0.2841 reply_top=1 pawn_convert=0.82 |
| 52 | t0371 | 0.667 | 0.533 | 1/2/0 | 0.61 | 4.94 | tactic_depth=4 phase_values=1.000,2.462,2.273,5.772,9.655;1.000,2.462,2.273,5.772,9.655;1.000,2.462,2.273,5.772,9.655 king_weight=0.02997 simplify_margin=1.615 mate_defence=1 |
| 53 | t0121 | 0.667 | 0.533 | 1/2/0 | 0.62 | 5.02 | tactic_rounds=4 mate_attack=3 simplify_weight=0.04056 opening_weight=0.5051 simplify_margin=0.1409 tactic_depth=4 ahead_value=0.5 exchange_depth=7 ahead_material=1 |
| 54 | t0398 | 0.667 | 0.533 | 2/0/1 | 0.63 | 5.23 | pawn_convert=0.7124 tactic_rounds=4 simplify_weight=0.14 simplify_margin=0.6461 tactic_top=6 scan_depth=1 opening_weight=0.2302 |
| 55 | t0195 | 0.667 | 0.533 | 2/0/1 | 0.64 | 5.24 | tactic_depth=4 pawn_weight=0.267 simplify_weight=0.4011 fifty_horizon=1 reply_top=1 pawn_convert=0.8141 ahead_value=0.7364 tactic_top=8 mate_attack=1 safety_weight=4.028 |
| 56 | t0333 | 0.667 | 0.533 | 2/0/1 | 0.65 | 5.13 | exchange_depth=7 scan_keep=9 tactic_nodes=165946 value_scaling=True tactic_depth=4 safety_weight=4.701 pawn_weight=0.4076 check_extension=1 |
| 57 | t0714 | 0.667 | 0.533 | 1/2/0 | 0.65 | 5.25 | repeat_as_draw=False scan_keep=14 exchange_depth=4 simplify_weight=0.01434 fifty_horizon=20 tactic_depth=5 |
| 58 | t0206 | 0.667 | 0.533 | 2/0/1 | 0.66 | 5.26 | value_scaling=True ahead_draws=True mate_attack=3 tactic_depth=4 reply_weight=0.09769 pawn_weight=1.581 |
| 59 | t0581 | 0.667 | 0.533 | 2/0/1 | 0.67 | 5.54 | exchange_depth=6 ahead_material=0.3743 scan_depth=0 phase_values=1.000,2.933,3.896,3.172,8.210;1.000,2.933,3.896,3.172,8.210;1.000,2.933,3.896,3.172,8.210 mate_defence=1 pawn_weight=0.2909 reply_weight=0.729 repeat_as_draw=False |
| 60 | t0077 | 0.667 | 0.533 | 1/2/0 | 0.68 | 5.53 | scan_depth=0 simplify_margin=0.1305 tactic_nodes=1353556 tactic_top=8 quiesce_promotions=False reply_weight=0.5936 |
| 61 | t0179 | 0.667 | 0.533 | 2/0/1 | 0.68 | 5.54 | quiesce_promotions=False scan_depth=1 pawn_weight=1.294 tactic_nodes=1500000 ahead_draws=True |
| 62 | t0718 | 0.667 | 0.533 | 2/0/1 | 0.68 | 5.40 | reply_top=1 tactic_top=4 mate_defence=1 safety_weight=1.557 exchange_depth=7 tactic_depth=5 ahead_value=0.9 |
| 63 | t0336 | 0.667 | 0.533 | 2/0/1 | 0.68 | 5.62 | ahead_value=0.5 mate_attack=1 quiesce_promotions=False safety_weight=1.697 scan_depth=1 pawn_convert=0.4586 value_scaling=True |
| 64 | t0190 | 0.667 | 0.533 | 1/2/0 | 0.69 | 5.69 | ahead_draws=True fifty_horizon=6 quiesce_promotions=False ahead_value=0.653 scan_keep=29 scan_depth=1 |
| 65 | t0005 | 0.667 | 0.533 | 2/0/1 | 0.71 | 7.51 | pawn_convert=1 fifty_horizon=30 simplify_weight=0.3801 scan_keep=3 ahead_value=0.7975 opening_weight=0.5923 |
| 66 | t0413 | 0.667 | 0.533 | 2/0/1 | 0.71 | 5.82 | mate_attack=3 reply_weight=0.5916 opening_weight=0.3325 pawn_weight=0.1881 scan_depth=0 |
| 67 | t0509 | 0.667 | 0.533 | 2/0/1 | 0.71 | 5.57 | safety_weight=6 reply_top=4 check_extension=1 scan_keep=28 reply_weight=0.7781 phase_values=1.000,3.098,2.206,3.634,8.756;1.000,3.098,2.206,3.634,8.756;1.000,3.098,2.206,3.634,8.756 scan_depth=2 tactic_depth=4 |
| 68 | t0224 | 0.667 | 0.533 | 1/2/0 | 0.73 | 5.94 | pawn_convert=0.5845 draw_value=0.2 tactic_rounds=4 opening_weight=0.3987 ahead_draws=True mate_attack=1 exchange_depth=7 value_scaling=True tactic_depth=5 repeat_as_draw=False |
| 69 | t0002 | 0.667 | 0.533 | 1/2/0 | 0.74 | 7.93 | simplify_weight=0.05445 fifty_horizon=26 pawn_weight=0.9151 repeat_as_draw=False quiesce_promotions=False tactic_top=6 |
| 70 | t0408 | 0.667 | 0.533 | 1/2/0 | 0.74 | 6.17 | repeat_as_draw=False scan_depth=0 simplify_weight=0.02645 scan_keep=24 ahead_value=0.9 pawn_convert=0.1887 |
| 71 | t0390 | 0.667 | 0.533 | 2/0/1 | 0.74 | 6.24 | tactic_depth=4 opening_weight=0.2901 check_extension=1 scan_keep=17 simplify_margin=1.275 |
| 72 | t0302 | 0.667 | 0.533 | 2/0/1 | 0.75 | 6.18 | pawn_convert=0.3461 ahead_draws=True quiesce_promotions=False scan_depth=2 value_scaling=True |
| 73 | t0501 | 0.667 | 0.533 | 2/0/1 | 0.76 | 6.12 | safety_weight=5.319 tactic_depth=5 simplify_weight=0.02989 ahead_value=0.5 simplify_margin=1.733 king_weight=0.1016 mate_attack=3 |
| 74 | t0291 | 0.667 | 0.533 | 1/2/0 | 0.76 | 5.80 | scan_keep=23 scan_depth=2 tactic_nodes=956567 pawn_weight=0.9918 ahead_draws=True simplify_weight=0.4861 safety_weight=3.714 simplify_margin=0.6293 draw_value=0.6 exchange_depth=7 |
| 75 | t0839 | 0.667 | 0.533 | 2/0/1 | 0.76 | 6.09 | repeat_as_draw=False tactic_depth=5 fifty_horizon=21 opening_weight=0.5088 simplify_margin=1.431 draw_value=0.6 tactic_nodes=262210 tactic_rounds=4 pawn_weight=0.9484 exchange_depth=7 |
| 76 | t0286 | 0.667 | 0.533 | 1/2/0 | 0.76 | 6.11 | reply_weight=0.4277 tactic_depth=5 opening_weight=0.1101 value_scaling=True simplify_margin=0.5323 |
| 77 | t0815 | 0.667 | 0.533 | 2/0/1 | 0.76 | 6.06 | mate_attack=1 simplify_margin=2.207 draw_value=0.2165 ahead_value=0.7371 scan_depth=2 mate_defence=1 ahead_material=0.9923 fifty_horizon=11 |
| 78 | t0393 | 0.667 | 0.533 | 2/0/1 | 0.76 | 6.44 | repeat_as_draw=False ahead_draws=True pawn_weight=0.2679 mate_attack=1 tactic_top=7 exchange_depth=6 draw_value=0.3888 scan_keep=4 quiesce_promotions=False |
| 79 | t0732 | 0.667 | 0.533 | 2/0/1 | 0.77 | 6.34 | fifty_horizon=20 tactic_top=11 tactic_nodes=1500000 repeat_as_draw=False quiesce_promotions=False scan_depth=0 simplify_margin=0.284 |
| 80 | t0424 | 0.667 | 0.533 | 2/0/1 | 0.78 | 6.51 | value_scaling=True reply_weight=0.8536 ahead_draws=True simplify_margin=0.2072 tactic_top=8 tactic_depth=5 |
| 81 | t0417 | 0.667 | 0.533 | 1/2/0 | 0.78 | 6.91 | mate_attack=1 fifty_horizon=29 repeat_as_draw=False quiesce_promotions=False ahead_value=0.5938 tactic_depth=5 value_scaling=True |
| 82 | t0006 | 0.667 | 0.533 | 2/0/1 | 0.78 | 8.18 | scan_keep=14 mate_defence=1 king_weight=0.01662 quiesce_promotions=False fifty_horizon=26 |
| 83 | t0226 | 0.667 | 0.533 | 1/2/0 | 0.78 | 6.27 | pawn_convert=0.2204 tactic_depth=4 fifty_horizon=17 ahead_material=0.3151 safety_weight=4.706 check_extension=1 quiesce_promotions=False king_weight=0.2962 simplify_weight=0.3617 |
| 84 | t0700 | 0.667 | 0.533 | 2/0/1 | 0.79 | 6.42 | check_extension=1 scan_keep=4 scan_depth=2 ahead_material=0.956 reply_weight=0.7444 simplify_margin=0.1503 pawn_convert=0.7 |
| 85 | t0438 | 0.667 | 0.533 | 2/0/1 | 0.80 | 6.67 | tactic_top=12 king_weight=0.06527 scan_depth=0 scan_keep=24 ahead_value=0.9 simplify_weight=0.2921 fifty_horizon=14 ahead_material=1 |
| 86 | t0851 | 0.667 | 0.533 | 1/2/0 | 0.80 | 6.45 | fifty_horizon=11 tactic_depth=4 check_extension=1 draw_value=0.6 simplify_weight=0.04513 tactic_nodes=840558 mate_attack=1 |
| 87 | t0018 | 0.667 | 0.533 | 2/0/1 | 0.80 | 6.63 | tactic_nodes=1225430 value_scaling=True mate_defence=1 ahead_material=1 scan_depth=0 opening_weight=0.02651 pawn_weight=0.7771 |
| 88 | t0974 | 0.667 | 0.533 | 2/0/1 | 0.83 | 6.85 | fifty_horizon=30 scan_depth=2 mate_attack=3 tactic_top=13 ahead_draws=True safety_weight=2.839 scan_keep=25 draw_value=0.3316 simplify_weight=0.2602 |
| 89 | t0652 | 0.667 | 0.533 | 1/2/0 | 0.84 | 6.80 | mate_defence=1 pawn_weight=0.2647 check_extension=1 ahead_draws=True tactic_top=6 mate_attack=3 scan_depth=0 tactic_nodes=815246 tactic_rounds=4 |
| 90 | t0122 | 0.667 | 0.533 | 1/2/0 | 0.84 | 7.09 | simplify_weight=0.1099 draw_value=0.3784 scan_keep=6 fifty_horizon=21 tactic_depth=5 quiesce_promotions=False value_scaling=True exchange_depth=3 |
| 91 | t0258 | 0.667 | 0.533 | 2/0/1 | 0.87 | 7.31 | tactic_top=11 simplify_margin=0.3353 ahead_material=0.1257 quiesce_promotions=False scan_keep=6 ahead_draws=True |
| 92 | t0720 | 0.667 | 0.533 | 1/2/0 | 0.87 | 7.25 | scan_keep=17 check_extension=1 scan_depth=2 fifty_horizon=9 king_weight=0.0405 ahead_material=0.253 tactic_rounds=4 tactic_nodes=438850 |
| 93 | t0278 | 0.667 | 0.533 | 1/2/0 | 0.87 | 6.98 | king_weight=0.1787 fifty_horizon=14 pawn_weight=1.27 exchange_depth=6 mate_attack=1 repeat_as_draw=False reply_weight=0.2767 phase_values=1.000,1.705,3.306,3.832,12.596;1.000,1.705,3.306,3.832,12.596;1.000,1.705,3.306,3.832,12.596 tactic_top=6 |
| 94 | t0066 | 0.667 | 0.533 | 1/2/0 | 0.87 | 7.14 | scan_keep=8 safety_weight=3.191 scan_depth=0 check_extension=1 simplify_margin=2.211 ahead_value=0.6954 opening_weight=0.03168 phase_values=1.000,2.077,3.383,4.801,12.333;1.000,2.077,3.383,4.801,12.333;1.000,2.077,3.383,4.801,12.333 value_scaling=True |
| 95 | t0556 | 0.667 | 0.533 | 2/0/1 | 0.87 | 7.08 | tactic_nodes=1500000 exchange_depth=6 fifty_horizon=0 safety_weight=6 reply_top=1 ahead_value=0.5 |
| 96 | t0181 | 0.667 | 0.533 | 1/2/0 | 0.88 | 6.97 | opening_weight=0.3108 scan_depth=2 check_extension=1 pawn_convert=1 reply_weight=0.1655 repeat_as_draw=False pawn_weight=0.3208 mate_defence=1 exchange_depth=4 |
| 97 | t0303 | 0.667 | 0.533 | 2/0/1 | 0.88 | 7.18 | quiesce_promotions=False opening_weight=0.2875 reply_top=2 fifty_horizon=1 tactic_depth=5 reply_weight=0.4543 pawn_weight=0.7307 pawn_convert=0.2506 |
| 98 | t0576 | 0.667 | 0.533 | 2/0/1 | 0.88 | 7.02 | safety_weight=2.883 pawn_weight=0.2732 ahead_draws=True check_extension=1 mate_attack=1 repeat_as_draw=False scan_depth=1 simplify_margin=0.485 |
| 99 | t0080 | 0.667 | 0.533 | 2/0/1 | 0.88 | 7.49 | tactic_top=14 scan_keep=19 scan_depth=2 value_scaling=True draw_value=0.6 check_extension=1 exchange_depth=7 |
| 100 | t0306 | 0.667 | 0.533 | 1/2/0 | 0.89 | 7.04 | fifty_horizon=23 king_weight=0.1092 pawn_convert=0.9591 ahead_material=0.9011 ahead_value=0.5 phase_values=1.000,2.882,3.792,3.333,7.670;1.000,2.882,3.792,3.333,7.670;1.000,2.882,3.792,3.333,7.670 scan_keep=11 check_extension=1 scan_depth=1 |
| 101 | t0213 | 0.667 | 0.533 | 1/2/0 | 0.89 | 7.13 | simplify_weight=0.01847 check_extension=1 ahead_draws=True tactic_depth=5 draw_value=0.4733 pawn_convert=0.1678 mate_attack=3 tactic_top=7 |
| 102 | t0886 | 0.667 | 0.533 | 1/2/0 | 0.89 | 7.72 | mate_defence=1 scan_keep=18 safety_weight=4.339 pawn_weight=0.762 tactic_rounds=2 phase_values=1.000,2.348,2.268,4.266,11.714;1.000,2.348,2.268,4.266,11.714;1.000,2.348,2.268,4.266,11.714 tactic_top=5 reply_weight=0.9363 repeat_as_draw=False simplify_margin=0.04824 |
| 103 | t0098 | 0.667 | 0.533 | 2/0/1 | 0.89 | 7.17 | scan_depth=2 safety_weight=2.719 pawn_weight=0.3084 check_extension=1 ahead_draws=True mate_defence=1 |
| 104 | t1014 | 0.667 | 0.533 | 2/0/1 | 0.90 | 7.34 | tactic_nodes=205468 check_extension=1 opening_weight=0.02984 mate_defence=1 ahead_draws=True scan_depth=1 ahead_value=0.616 |
| 105 | t0376 | 0.667 | 0.533 | 2/0/1 | 0.90 | 7.31 | fifty_horizon=25 value_scaling=True exchange_depth=4 simplify_weight=0.08512 ahead_material=0.01742 |
| 106 | t0523 | 0.667 | 0.533 | 2/0/1 | 0.91 | 7.21 | reply_weight=0.5657 tactic_top=6 king_weight=0.4429 mate_defence=3 draw_value=0.6 mate_attack=3 opening_weight=0.08012 ahead_value=0.6738 exchange_depth=4 |
| 107 | t0884 | 0.667 | 0.533 | 2/0/1 | 0.92 | 7.76 | pawn_weight=0.5261 repeat_as_draw=False ahead_draws=True pawn_convert=0.3263 ahead_material=0.13 phase_values=1.000,2.265,3.833,3.741,6.713;1.000,2.265,3.833,3.741,6.713;1.000,2.265,3.833,3.741,6.713 mate_defence=1 exchange_depth=6 value_scaling=True |
| 108 | t0504 | 0.667 | 0.533 | 1/2/0 | 0.92 | 7.52 | quiesce_promotions=False reply_weight=0.07543 mate_attack=3 fifty_horizon=0 exchange_depth=7 tactic_rounds=4 king_weight=0.09215 |
| 109 | t0313 | 0.667 | 0.533 | 1/2/0 | 0.93 | 7.53 | ahead_value=0.9 king_weight=0.2436 ahead_draws=True exchange_depth=7 pawn_convert=0.2592 simplify_weight=0.4028 mate_attack=3 |
| 110 | t0201 | 0.667 | 0.533 | 1/2/0 | 0.93 | 7.35 | tactic_nodes=597040 pawn_convert=0.1615 tactic_top=7 exchange_depth=3 ahead_value=0.5741 pawn_weight=0.3118 ahead_draws=True fifty_horizon=20 scan_depth=2 ahead_material=0.4149 |
| 111 | t0606 | 0.667 | 0.533 | 1/2/0 | 0.94 | 7.33 | value_scaling=True safety_weight=2.057 phase_values=1.000,2.602,2.304,3.585,13.242;1.000,2.602,2.304,3.585,13.242;1.000,2.602,2.304,3.585,13.242 reply_top=3 draw_value=0.2 tactic_depth=4 pawn_weight=0.2703 check_extension=1 pawn_convert=1 tactic_top=12 |
| 112 | t0081 | 0.667 | 0.533 | 1/2/0 | 0.94 | 8.00 | pawn_weight=1.059 reply_weight=0.8561 tactic_nodes=1500000 mate_attack=3 draw_value=0.3026 |
| 113 | t0290 | 0.667 | 0.533 | 2/0/1 | 0.94 | 7.71 | pawn_convert=0.3174 ahead_value=0.6994 simplify_margin=0.1272 reply_weight=0.8997 pawn_weight=1.396 quiesce_promotions=False |
| 114 | t0370 | 0.667 | 0.533 | 1/2/0 | 0.94 | 7.71 | ahead_draws=True opening_weight=0.3803 draw_value=0.3021 scan_keep=14 simplify_weight=0.3914 quiesce_promotions=False tactic_top=5 ahead_material=0.02317 value_scaling=True phase_values=1.000,1.960,2.375,2.866,7.725;1.000,1.960,2.375,2.866,7.725;1.000,1.960,2.375,2.866,7.725 |
| 115 | t0036 | 0.667 | 0.533 | 2/0/1 | 0.95 | 7.98 | scan_keep=30 scan_depth=0 opening_weight=0.1665 ahead_material=0.114 safety_weight=4.417 mate_attack=1 fifty_horizon=24 tactic_nodes=379417 check_extension=1 |
| 116 | t1021 | 0.667 | 0.533 | 2/0/1 | 0.95 | 6.60 | pawn_convert=0.01633 ahead_draws=True ahead_value=0.7205 value_scaling=True ahead_material=0.1618 simplify_margin=0.1171 |
| 117 | t0034 | 0.667 | 0.533 | 2/0/1 | 0.96 | 7.81 | simplify_weight=0.04833 simplify_margin=0.1494 mate_defence=1 tactic_depth=5 reply_top=4 tactic_nodes=1500000 ahead_value=0.7189 |
| 118 | t0051 | 0.667 | 0.533 | 2/0/1 | 0.97 | 7.83 | scan_depth=2 tactic_top=12 scan_keep=0 tactic_rounds=2 check_extension=1 pawn_convert=0.4096 ahead_value=0.8186 |
| 119 | t0188 | 0.667 | 0.533 | 2/0/1 | 0.97 | 7.69 | simplify_margin=1.726 tactic_top=6 pawn_convert=0.6717 king_weight=0.2728 ahead_value=0.5 |
| 120 | t0479 | 0.667 | 0.533 | 2/0/1 | 0.98 | 7.74 | pawn_convert=0.5115 exchange_depth=7 tactic_nodes=482097 phase_values=1.000,2.852,2.406,4.321,6.539;1.000,2.852,2.406,4.321,6.539;1.000,2.852,2.406,4.321,6.539 safety_weight=1.389 pawn_weight=1.635 draw_value=0.5786 |
| 121 | t0490 | 0.667 | 0.533 | 1/2/0 | 0.98 | 7.92 | simplify_weight=0.06204 mate_attack=1 tactic_top=8 draw_value=0.6 safety_weight=6 |
| 122 | t0458 | 0.667 | 0.533 | 2/0/1 | 0.98 | 8.07 | fifty_horizon=12 exchange_depth=7 king_weight=0.04931 ahead_draws=True safety_weight=1.857 tactic_rounds=2 pawn_convert=0.2895 mate_defence=3 ahead_value=0.9 |
| 123 | t0558 | 0.667 | 0.533 | 2/0/1 | 0.98 | 7.93 | tactic_top=7 pawn_convert=0.6669 safety_weight=2.558 mate_attack=1 value_scaling=True tactic_nodes=1500000 ahead_draws=True fifty_horizon=8 king_weight=0.01457 draw_value=0.2301 |
| 124 | t0145 | 0.667 | 0.533 | 2/0/1 | 0.99 | 7.93 | tactic_rounds=4 king_weight=0.3966 ahead_draws=True mate_attack=1 fifty_horizon=19 reply_top=1 pawn_convert=1 |
| 125 | t0344 | 0.667 | 0.533 | 1/2/0 | 0.99 | 7.93 | king_weight=0.3809 opening_weight=0.579 pawn_weight=0.1665 ahead_value=0.9 quiesce_promotions=False draw_value=0.6 |
| 126 | t0014 | 0.667 | 0.533 | 1/2/0 | 0.99 | 8.15 | check_extension=1 ahead_material=0.5966 king_weight=0.0494 tactic_depth=5 reply_weight=0.3567 tactic_top=7 |
| 127 | t0083 | 0.667 | 0.533 | 1/2/0 | 0.99 | 8.19 | tactic_rounds=2 pawn_weight=0.6631 scan_keep=25 draw_value=0.6 reply_top=1 |
| 128 | t0515 | 0.667 | 0.533 | 1/2/0 | 0.99 | 7.95 | mate_attack=1 quiesce_promotions=False ahead_value=0.5 reply_weight=0.7696 value_scaling=True king_weight=0.03562 pawn_weight=0.5607 |
| 129 | t0653 | 0.667 | 0.533 | 1/2/0 | 1.00 | 8.19 | mate_defence=3 fifty_horizon=15 exchange_depth=7 repeat_as_draw=False scan_keep=24 ahead_draws=True tactic_top=14 tactic_rounds=2 reply_top=1 |
| 130 | t0760 | 0.667 | 0.533 | 1/2/0 | 1.00 | 8.25 | simplify_margin=1.408 tactic_rounds=2 pawn_weight=0.4696 king_weight=0.3099 safety_weight=6 repeat_as_draw=False ahead_draws=True |
| 131 | t0446 | 0.667 | 0.533 | 1/2/0 | 1.00 | 8.47 | quiesce_promotions=False draw_value=0.2941 king_weight=0.06738 tactic_top=8 simplify_weight=0.1077 mate_attack=3 simplify_margin=0.7872 pawn_convert=0.9365 |
| 132 | t0753 | 0.500 | 0.566 | 1/1/1 | 0.37 | 3.08 | ahead_draws=True exchange_depth=4 reply_top=1 scan_depth=2 king_weight=0.1206 tactic_depth=4 pawn_weight=0.4902 reply_weight=0.1922 |
| 133 | t0524 | 0.500 | 0.566 | 1/1/1 | 0.38 | 3.07 | ahead_draws=True tactic_depth=5 scan_depth=1 exchange_depth=4 quiesce_promotions=False tactic_rounds=1 |
| 134 | t0623 | 0.500 | 0.566 | 1/1/1 | 0.47 | 3.81 | simplify_margin=2.196 ahead_material=0.3531 safety_weight=4.229 check_extension=1 quiesce_promotions=False tactic_depth=4 scan_depth=2 mate_defence=3 |
| 135 | t0275 | 0.500 | 0.566 | 1/1/1 | 0.52 | 4.03 | opening_weight=0.2185 draw_value=0.6 quiesce_promotions=False reply_weight=0.1955 repeat_as_draw=False tactic_depth=4 check_extension=1 reply_top=4 scan_depth=0 |
| 136 | t0687 | 0.500 | 0.566 | 1/1/1 | 0.54 | 4.14 | pawn_weight=0.5557 draw_value=0.3175 scan_depth=2 value_scaling=True tactic_depth=5 reply_weight=0.2328 ahead_draws=True safety_weight=5.428 repeat_as_draw=False |
| 137 | t0170 | 0.500 | 0.566 | 0/3/0 | 0.54 | 4.42 | fifty_horizon=22 repeat_as_draw=False value_scaling=True scan_depth=2 check_extension=1 scan_keep=9 mate_attack=1 tactic_rounds=4 tactic_depth=5 |
| 138 | t0853 | 0.500 | 0.566 | 1/1/1 | 0.56 | 4.45 | scan_depth=2 mate_attack=3 tactic_depth=5 king_weight=0.1985 safety_weight=4.974 opening_weight=0.0578 ahead_material=0.4851 mate_defence=3 tactic_rounds=1 pawn_convert=0.1655 |
| 139 | t0113 | 0.500 | 0.566 | 1/1/1 | 0.57 | 4.51 | tactic_top=8 opening_weight=0.04427 tactic_nodes=1010123 tactic_depth=5 reply_weight=1 check_extension=1 scan_depth=2 quiesce_promotions=False exchange_depth=6 |
| 140 | t0144 | 0.500 | 0.566 | 1/1/1 | 0.60 | 4.94 | phase_values=1.000,1.928,2.043,4.156,7.397;1.000,1.928,2.043,4.156,7.397;1.000,1.928,2.043,4.156,7.397 reply_weight=0.3953 exchange_depth=6 tactic_depth=4 mate_attack=1 tactic_nodes=1128329 reply_top=1 |
| 141 | t0447 | 0.500 | 0.566 | 1/1/1 | 0.61 | 5.69 | tactic_nodes=643859 scan_keep=17 pawn_convert=0.1835 draw_value=0.3618 safety_weight=2.147 scan_depth=1 ahead_material=1 reply_weight=0.214 |
| 142 | t0205 | 0.500 | 0.566 | 1/1/1 | 0.62 | 5.09 | ahead_draws=True tactic_depth=4 draw_value=0.4334 pawn_convert=0.1947 mate_attack=3 simplify_weight=0.2647 |
| 143 | t0243 | 0.500 | 0.566 | 1/1/1 | 0.63 | 5.12 | tactic_rounds=2 quiesce_promotions=False ahead_draws=True pawn_weight=1.109 simplify_weight=0.02056 value_scaling=True safety_weight=1.839 pawn_convert=0.318 tactic_top=4 scan_depth=2 |
| 144 | t0249 | 0.500 | 0.566 | 1/1/1 | 0.63 | 4.90 | phase_values=1.000,1.911,3.418,3.218,7.115;1.000,1.911,3.418,3.218,7.115;1.000,1.911,3.418,3.218,7.115 draw_value=0.3815 simplify_weight=0.2492 exchange_depth=6 tactic_depth=4 ahead_draws=True tactic_top=6 |
| 145 | t0064 | 0.500 | 0.566 | 1/1/1 | 0.63 | 5.12 | opening_weight=0.4485 reply_weight=0.1011 ahead_material=0.8273 simplify_margin=0.2095 safety_weight=4.534 tactic_rounds=2 value_scaling=True simplify_weight=0.06756 tactic_depth=4 quiesce_promotions=False |
| 146 | t0541 | 0.500 | 0.566 | 1/1/1 | 0.65 | 5.07 | scan_depth=1 opening_weight=0.0239 quiesce_promotions=False pawn_weight=1.034 repeat_as_draw=False exchange_depth=6 |
| 147 | t0119 | 0.500 | 0.566 | 1/1/1 | 0.66 | 5.34 | pawn_convert=0.7252 ahead_material=0.2774 safety_weight=1.571 tactic_depth=4 king_weight=0.09949 |
| 148 | t0722 | 0.500 | 0.566 | 1/1/1 | 0.67 | 5.26 | reply_weight=0.6274 value_scaling=True repeat_as_draw=False simplify_margin=1.498 tactic_depth=4 |
| 149 | t0706 | 0.500 | 0.566 | 1/1/1 | 0.68 | 5.64 | tactic_depth=4 tactic_top=12 scan_keep=12 ahead_draws=True fifty_horizon=27 safety_weight=1.352 reply_top=2 repeat_as_draw=False exchange_depth=6 opening_weight=0.165 |
| 150 | t0962 | 0.500 | 0.566 | 0/3/0 | 0.68 | 5.74 | mate_attack=3 ahead_draws=True tactic_nodes=1500000 scan_depth=1 pawn_weight=1.808 tactic_top=9 pawn_convert=0.09936 |
| 151 | t0241 | 0.500 | 0.566 | 1/1/1 | 0.68 | 5.59 | tactic_nodes=369070 safety_weight=3.365 mate_defence=3 tactic_depth=4 ahead_material=0.1649 reply_weight=0.08908 tactic_rounds=2 ahead_value=0.6949 |
| 152 | t0963 | 0.500 | 0.566 | 1/1/1 | 0.68 | 5.70 | safety_weight=1.726 opening_weight=0.2278 tactic_nodes=950876 pawn_weight=1.209 scan_depth=1 pawn_convert=0.7538 |
| 153 | t0372 | 0.500 | 0.566 | 1/1/1 | 0.68 | 5.56 | quiesce_promotions=False ahead_value=0.5 draw_value=0.6 tactic_depth=4 mate_defence=3 ahead_material=0.4034 |
| 154 | t0918 | 0.500 | 0.566 | 0/3/0 | 0.69 | 5.68 | simplify_weight=0.05641 pawn_convert=0.6749 scan_depth=1 ahead_value=0.9 tactic_rounds=2 opening_weight=0.1756 |
| 155 | t0130 | 0.500 | 0.566 | 1/1/1 | 0.71 | 5.82 | ahead_value=0.6194 opening_weight=0.3494 king_weight=0.2843 reply_weight=0.2048 scan_depth=0 mate_attack=1 |
| 156 | t0107 | 0.500 | 0.566 | 1/1/1 | 0.73 | 5.85 | tactic_nodes=914016 safety_weight=2.986 fifty_horizon=3 repeat_as_draw=False scan_depth=2 pawn_weight=0.5259 simplify_margin=0.1605 tactic_top=9 |
| 157 | t0124 | 0.500 | 0.566 | 1/1/1 | 0.74 | 6.08 | simplify_margin=0.154 tactic_top=9 tactic_rounds=2 pawn_convert=0.9126 reply_weight=0.1972 exchange_depth=4 scan_depth=0 reply_top=2 |
| 158 | t0595 | 0.500 | 0.566 | 1/1/1 | 0.74 | 6.14 | scan_depth=0 mate_attack=3 scan_keep=4 value_scaling=True ahead_material=0.632 |
| 159 | t0468 | 0.500 | 0.566 | 1/1/1 | 0.77 | 6.29 | check_extension=1 opening_weight=0.3707 tactic_depth=4 value_scaling=True tactic_rounds=4 |
| 160 | t0263 | 0.500 | 0.566 | 1/1/1 | 0.77 | 6.18 | scan_keep=23 tactic_top=6 exchange_depth=6 repeat_as_draw=False fifty_horizon=25 mate_attack=3 reply_top=3 scan_depth=0 ahead_material=1 quiesce_promotions=False |
| 161 | t0102 | 0.500 | 0.566 | 1/1/1 | 0.77 | 6.20 | value_scaling=True tactic_depth=4 tactic_top=12 fifty_horizon=21 king_weight=0.03369 repeat_as_draw=False tactic_nodes=229759 check_extension=1 |
| 162 | t0896 | 0.500 | 0.566 | 1/1/1 | 0.78 | 6.44 | mate_attack=3 fifty_horizon=19 tactic_depth=5 draw_value=0.3398 reply_weight=0.08481 simplify_margin=1.567 tactic_nodes=648822 quiesce_promotions=False |
| 163 | t0453 | 0.500 | 0.566 | 1/1/1 | 0.79 | 6.61 | exchange_depth=4 tactic_nodes=366919 scan_keep=0 pawn_convert=0.6935 ahead_draws=True ahead_material=0.1138 simplify_weight=0.08052 reply_weight=0.5534 fifty_horizon=0 scan_depth=2 |
| 164 | t0643 | 0.500 | 0.566 | 1/1/1 | 0.79 | 6.24 | mate_defence=3 value_scaling=True scan_depth=1 mate_attack=3 opening_weight=0.222 simplify_margin=1.057 |
| 165 | t0212 | 0.500 | 0.566 | 1/1/1 | 0.80 | 6.39 | reply_weight=0.1131 king_weight=0.4922 simplify_margin=1.425 repeat_as_draw=False ahead_draws=True simplify_weight=0.4414 opening_weight=0.3406 tactic_depth=4 check_extension=1 |
| 166 | t0719 | 0.500 | 0.566 | 1/1/1 | 0.80 | 6.24 | quiesce_promotions=False king_weight=0.3103 opening_weight=0.5566 tactic_rounds=1 tactic_top=11 scan_depth=2 pawn_convert=0.7823 ahead_value=0.6333 |
| 167 | t0090 | 0.500 | 0.566 | 1/1/1 | 0.80 | 6.56 | quiesce_promotions=False king_weight=0.02586 scan_depth=2 phase_values=1.000,1.790,2.341,3.821,8.904;1.000,1.790,2.341,3.821,8.904;1.000,1.790,2.341,3.821,8.904 opening_weight=0.3511 |
| 168 | t0579 | 0.500 | 0.566 | 1/1/1 | 0.81 | 6.61 | pawn_weight=1.734 mate_defence=1 scan_depth=0 simplify_margin=0.3275 exchange_depth=7 tactic_nodes=1500000 check_extension=1 reply_weight=0.1332 opening_weight=0.1094 |
| 169 | t0445 | 0.500 | 0.566 | 1/1/1 | 0.82 | 6.85 | ahead_material=0.3672 ahead_draws=True value_scaling=True pawn_convert=0.208 fifty_horizon=21 quiesce_promotions=False scan_depth=0 tactic_top=14 |
| 170 | t0590 | 0.500 | 0.566 | 0/3/0 | 0.84 | 6.93 | ahead_material=0.5619 scan_depth=0 draw_value=0.2 reply_top=3 exchange_depth=4 repeat_as_draw=False safety_weight=3.165 quiesce_promotions=False |
| 171 | t0048 | 0.500 | 0.566 | 1/1/1 | 0.84 | 6.78 | tactic_depth=5 ahead_draws=True exchange_depth=6 reply_top=2 value_scaling=True reply_weight=1 ahead_value=0.7544 opening_weight=0.05046 repeat_as_draw=False ahead_material=0.1402 |
| 172 | t0588 | 0.500 | 0.566 | 0/3/0 | 0.85 | 6.90 | mate_attack=1 exchange_depth=7 reply_top=4 quiesce_promotions=False value_scaling=True tactic_top=12 tactic_depth=4 reply_weight=0.07935 |
| 173 | t0406 | 0.500 | 0.566 | 1/1/1 | 0.86 | 7.83 | tactic_rounds=1 pawn_weight=0.232 simplify_weight=0.01625 quiesce_promotions=False tactic_top=13 simplify_margin=2.157 exchange_depth=6 |
| 174 | t0475 | 0.500 | 0.566 | 1/1/1 | 0.87 | 6.92 | scan_depth=2 value_scaling=True safety_weight=4.102 simplify_weight=0.0202 quiesce_promotions=False draw_value=0.5341 ahead_value=0.6363 check_extension=1 |
| 175 | t0551 | 0.500 | 0.566 | 1/1/1 | 0.87 | 7.10 | quiesce_promotions=False tactic_nodes=718573 reply_top=2 scan_keep=23 simplify_margin=0.8861 tactic_depth=5 exchange_depth=7 tactic_rounds=4 king_weight=0.243 |
| 176 | t0795 | 0.500 | 0.566 | 1/1/1 | 0.87 | 7.14 | pawn_weight=1.314 scan_depth=1 check_extension=1 quiesce_promotions=False king_weight=0.08235 safety_weight=5.366 |
| 177 | t0593 | 0.500 | 0.566 | 1/1/1 | 0.87 | 7.16 | safety_weight=1.611 tactic_top=4 draw_value=0.209 pawn_weight=0.07202 ahead_draws=True exchange_depth=7 tactic_nodes=325691 tactic_rounds=1 mate_defence=3 |
| 178 | t0630 | 0.500 | 0.566 | 1/1/1 | 0.88 | 7.29 | tactic_top=12 mate_attack=3 scan_keep=23 fifty_horizon=3 king_weight=0.1447 exchange_depth=4 ahead_draws=True reply_weight=0.09425 scan_depth=0 reply_top=3 |
| 179 | t0162 | 0.500 | 0.566 | 1/1/1 | 0.88 | 7.05 | tactic_top=7 mate_attack=1 simplify_margin=1.219 scan_depth=2 check_extension=1 ahead_material=0.9318 draw_value=0.3616 |
| 180 | t0702 | 0.500 | 0.566 | 1/1/1 | 0.89 | 7.09 | simplify_weight=0.4597 tactic_depth=4 value_scaling=True ahead_material=0.2706 reply_top=2 pawn_weight=0.2252 check_extension=1 |
| 181 | t0118 | 0.500 | 0.566 | 1/1/1 | 0.90 | 7.40 | pawn_weight=0.3154 ahead_material=0.8631 reply_top=1 simplify_margin=2.036 simplify_weight=0.01642 scan_keep=6 ahead_value=0.9 |
| 182 | t0789 | 0.500 | 0.566 | 1/1/1 | 0.91 | 7.27 | simplify_margin=1.861 tactic_top=9 scan_keep=0 mate_attack=1 phase_values=1.000,2.687,2.132,4.302,7.032;1.000,2.687,2.132,4.302,7.032;1.000,2.687,2.132,4.302,7.032 pawn_convert=0.9849 exchange_depth=4 pawn_weight=1.728 tactic_rounds=4 |
| 183 | t0003 | 0.500 | 0.566 | 1/1/1 | 0.91 | 9.58 | ahead_draws=True scan_keep=12 mate_attack=3 repeat_as_draw=False reply_top=3 simplify_weight=0.262 |
| 184 | t0587 | 0.500 | 0.566 | 1/1/1 | 0.91 | 7.38 | value_scaling=True ahead_material=0.3165 exchange_depth=7 mate_defence=1 king_weight=0.08399 |
| 185 | t0125 | 0.500 | 0.566 | 1/1/1 | 0.92 | 7.72 | phase_values=1.000,2.133,3.599,3.675,7.802;1.000,2.133,3.599,3.675,7.802;1.000,2.133,3.599,3.675,7.802 reply_weight=0.2195 simplify_margin=0.4236 opening_weight=0.06077 draw_value=0.3237 |
| 186 | t0736 | 0.500 | 0.566 | 1/1/1 | 0.92 | 7.38 | reply_weight=0.1555 mate_attack=3 quiesce_promotions=False scan_keep=10 tactic_top=6 tactic_rounds=4 repeat_as_draw=False |
| 187 | t0548 | 0.500 | 0.566 | 1/1/1 | 0.92 | 7.37 | tactic_rounds=1 scan_depth=0 phase_values=1.000,1.587,2.876,3.380,10.736;1.000,1.587,2.876,3.380,10.736;1.000,1.587,2.876,3.380,10.736 safety_weight=4.342 king_weight=0.05095 check_extension=1 simplify_weight=0.1652 repeat_as_draw=False fifty_horizon=30 |
| 188 | t0160 | 0.500 | 0.566 | 0/3/0 | 0.92 | 7.61 | ahead_value=0.5136 mate_defence=3 reply_top=2 repeat_as_draw=False draw_value=0.4707 reply_weight=0.6092 scan_depth=2 |
| 189 | t0505 | 0.500 | 0.566 | 1/1/1 | 0.92 | 7.65 | tactic_nodes=499353 tactic_depth=5 mate_defence=1 ahead_value=0.7715 fifty_horizon=13 opening_weight=0.06186 simplify_margin=0.03396 reply_top=3 |
| 190 | t0707 | 0.500 | 0.566 | 1/1/1 | 0.92 | 7.56 | tactic_nodes=616504 draw_value=0.2952 ahead_material=0.9978 king_weight=0.03383 reply_top=1 ahead_value=0.7047 ahead_draws=True scan_keep=7 mate_defence=1 |
| 191 | t0123 | 0.500 | 0.566 | 1/1/1 | 0.93 | 7.63 | tactic_nodes=1231008 fifty_horizon=21 tactic_rounds=2 ahead_value=0.5 tactic_top=6 phase_values=1.000,2.702,2.936,4.009,9.215;1.000,2.702,2.936,4.009,9.215;1.000,2.702,2.936,4.009,9.215 ahead_material=1 quiesce_promotions=False |
| 192 | t0035 | 0.500 | 0.566 | 1/1/1 | 0.93 | 8.03 | quiesce_promotions=False tactic_rounds=4 pawn_convert=0.479 scan_keep=15 ahead_material=0.9906 reply_top=1 |
| 193 | t0198 | 0.500 | 0.566 | 1/1/1 | 0.93 | 7.55 | exchange_depth=6 opening_weight=0.1414 pawn_weight=0.2265 fifty_horizon=15 ahead_value=0.6837 tactic_rounds=2 check_extension=1 mate_attack=1 simplify_margin=0.4654 tactic_depth=5 |
| 194 | t0642 | 0.500 | 0.566 | 1/1/1 | 0.94 | 7.58 | repeat_as_draw=False pawn_convert=0.7003 king_weight=0.1933 mate_attack=1 ahead_draws=True scan_depth=0 tactic_top=12 ahead_value=0.7976 phase_values=1.000,1.631,1.874,5.686,6.641;1.000,1.631,1.874,5.686,6.641;1.000,1.631,1.874,5.686,6.641 |
| 195 | t0421 | 0.500 | 0.566 | 1/1/1 | 0.95 | 7.95 | ahead_draws=True tactic_depth=5 mate_attack=3 ahead_value=0.888 tactic_nodes=178595 phase_values=1.000,2.277,2.444,3.741,6.744;1.000,2.277,2.444,3.741,6.744;1.000,2.277,2.444,3.741,6.744 pawn_weight=0.2507 reply_top=3 |
| 196 | t0467 | 0.500 | 0.566 | 1/1/1 | 0.95 | 7.68 | scan_keep=12 value_scaling=True reply_weight=0.2404 fifty_horizon=28 safety_weight=3.416 |
| 197 | t0007 | 0.500 | 0.566 | 1/1/1 | 0.95 | 9.97 | mate_attack=1 draw_value=0.4582 exchange_depth=4 opening_weight=0.05612 ahead_draws=True pawn_weight=0.3317 safety_weight=5.9 check_extension=1 tactic_top=8 |
| 198 | t0735 | 0.500 | 0.566 | 1/1/1 | 0.95 | 7.72 | phase_values=1.000,2.392,2.078,3.817,7.775;1.000,2.392,2.078,3.817,7.775;1.000,2.392,2.078,3.817,7.775 quiesce_promotions=False tactic_rounds=2 opening_weight=0.03309 ahead_draws=True king_weight=0.1721 fifty_horizon=26 |
| 199 | t0657 | 0.500 | 0.566 | 1/1/1 | 0.95 | 7.30 | draw_value=0.2569 scan_depth=2 fifty_horizon=13 check_extension=1 tactic_rounds=1 |
| 200 | t0699 | 0.500 | 0.566 | 0/3/0 | 0.95 | 7.73 | pawn_convert=0.1442 repeat_as_draw=False phase_values=1.000,2.365,3.489,2.892,7.952;1.000,2.365,3.489,2.892,7.952;1.000,2.365,3.489,2.892,7.952 simplify_weight=0.4428 quiesce_promotions=False |
| 201 | t0607 | 0.500 | 0.566 | 1/1/1 | 0.95 | 7.73 | draw_value=0.52 mate_defence=3 value_scaling=True exchange_depth=4 simplify_weight=0.02003 |
| 202 | t1002 | 0.500 | 0.566 | 1/1/1 | 0.96 | 8.11 | draw_value=0.565 mate_defence=1 phase_values=1.000,2.877,3.903,3.329,8.751;1.000,2.877,3.903,3.329,8.751;1.000,2.877,3.903,3.329,8.751 exchange_depth=7 simplify_weight=0.3566 quiesce_promotions=False ahead_draws=True |
| 203 | t0570 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.63 | pawn_convert=0.7742 check_extension=1 tactic_depth=4 opening_weight=0.1055 reply_top=3 pawn_weight=1.622 reply_weight=0.2058 phase_values=1.000,2.390,3.301,5.544,9.733;1.000,2.390,3.301,5.544,9.733;1.000,2.390,3.301,5.544,9.733 |
| 204 | t0751 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.62 | reply_weight=0.3993 phase_values=1.000,1.571,2.218,5.397,13.447;1.000,1.571,2.218,5.397,13.447;1.000,1.571,2.218,5.397,13.447 exchange_depth=6 fifty_horizon=23 ahead_material=0.9654 pawn_weight=0.05185 pawn_convert=1 mate_attack=1 |
| 205 | t0244 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.77 | reply_weight=0.1083 value_scaling=True repeat_as_draw=False ahead_draws=True draw_value=0.5137 pawn_convert=0.9187 mate_attack=3 phase_values=1.000,2.529,2.251,3.943,6.795;1.000,2.529,2.251,3.943,6.795;1.000,2.529,2.251,3.943,6.795 exchange_depth=7 |
| 206 | t0578 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.65 | value_scaling=True repeat_as_draw=False scan_keep=18 pawn_weight=0.6502 tactic_nodes=1500000 draw_value=0.2 king_weight=0.07411 phase_values=1.000,2.313,2.160,3.679,10.322;1.000,2.313,2.160,3.679,10.322;1.000,2.313,2.160,3.679,10.322 ahead_material=0.8053 exchange_depth=7 |
| 207 | t0111 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.62 | mate_attack=1 ahead_material=0.9927 simplify_margin=0.9805 pawn_convert=1 reply_top=1 exchange_depth=4 |
| 208 | t0288 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.55 | opening_weight=0.1898 tactic_top=7 ahead_draws=True tactic_rounds=1 quiesce_promotions=False repeat_as_draw=False fifty_horizon=7 |
| 209 | t0792 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.82 | mate_attack=3 safety_weight=5.137 scan_depth=2 tactic_nodes=155645 exchange_depth=3 phase_values=1.000,2.630,3.354,4.472,6.500;1.000,2.630,3.354,4.472,6.500;1.000,2.630,3.354,4.472,6.500 pawn_convert=0.8547 ahead_material=0.2868 tactic_rounds=2 |
| 210 | t0709 | 0.500 | 0.566 | 1/1/1 | 0.96 | 7.78 | reply_weight=0.1691 pawn_weight=1.69 phase_values=1.000,2.465,3.806,3.201,9.483;1.000,2.465,3.806,3.201,9.483;1.000,2.465,3.806,3.201,9.483 opening_weight=0.03089 exchange_depth=4 pawn_convert=0.9575 quiesce_promotions=False |
| 211 | t0019 | 0.500 | 0.566 | 0/3/0 | 0.97 | 8.17 | fifty_horizon=1 simplify_weight=0.3354 tactic_rounds=2 safety_weight=6 ahead_value=0.9 value_scaling=True repeat_as_draw=False draw_value=0.6 tactic_top=6 ahead_material=0.8441 |
| 212 | t0465 | 0.500 | 0.566 | 1/1/1 | 0.97 | 7.74 | scan_depth=0 value_scaling=True simplify_margin=0.257 ahead_draws=True check_extension=1 |
| 213 | t0058 | 0.500 | 0.566 | 1/1/1 | 0.97 | 7.31 | tactic_top=4 tactic_nodes=182368 mate_attack=1 fifty_horizon=1 simplify_weight=0.4219 ahead_material=0.4403 |
| 214 | t0674 | 0.500 | 0.566 | 1/1/1 | 0.97 | 7.58 | tactic_depth=5 simplify_weight=0.1653 ahead_draws=True tactic_nodes=687790 safety_weight=4.692 simplify_margin=0.3712 check_extension=1 ahead_material=0.2116 repeat_as_draw=False phase_values=1.000,2.345,3.294,5.269,6.691;1.000,2.345,3.294,5.269,6.691;1.000,2.345,3.294,5.269,6.691 |
| 215 | t0939 | 0.500 | 0.566 | 1/1/1 | 0.97 | 8.04 | fifty_horizon=23 ahead_material=0.02772 exchange_depth=7 simplify_weight=0.04869 pawn_weight=1.693 ahead_value=0.7274 reply_weight=0.2487 mate_defence=3 opening_weight=0.1192 repeat_as_draw=False |
| 216 | t0635 | 0.500 | 0.566 | 1/1/1 | 0.97 | 7.71 | draw_value=0.402 reply_weight=0.2198 ahead_value=0.6361 exchange_depth=7 tactic_nodes=198974 tactic_rounds=2 simplify_margin=0.832 |
| 217 | t0484 | 0.500 | 0.566 | 1/1/1 | 0.97 | 7.61 | tactic_rounds=1 opening_weight=0.3619 pawn_convert=0.2077 fifty_horizon=5 repeat_as_draw=False exchange_depth=4 ahead_value=0.5 value_scaling=True reply_top=3 scan_keep=2 |
| 218 | t0997 | 0.500 | 0.566 | 1/1/1 | 0.97 | 8.12 | simplify_weight=0.01444 safety_weight=2.236 ahead_draws=True opening_weight=0.03859 mate_attack=3 tactic_top=12 fifty_horizon=11 scan_keep=12 |
| 219 | t0065 | 0.500 | 0.566 | 1/1/1 | 0.98 | 7.96 | king_weight=0.06483 safety_weight=3.625 opening_weight=0.03764 reply_top=1 ahead_material=0.2336 |
| 220 | t0151 | 0.500 | 0.566 | 1/1/1 | 0.98 | 7.60 | pawn_convert=0.2373 tactic_rounds=2 mate_attack=3 fifty_horizon=14 safety_weight=2.339 phase_values=1.000,2.736,2.867,5.314,11.502;1.000,2.736,2.867,5.314,11.502;1.000,2.736,2.867,5.314,11.502 scan_keep=23 tactic_top=7 opening_weight=0.4896 |
| 221 | t0487 | 0.500 | 0.566 | 1/1/1 | 0.98 | 7.93 | mate_attack=3 reply_top=1 king_weight=0.03305 ahead_value=0.9 safety_weight=4.588 check_extension=1 repeat_as_draw=False pawn_weight=0.5038 scan_depth=0 simplify_margin=0.2876 |
| 222 | t0184 | 0.500 | 0.566 | 1/1/1 | 0.98 | 7.90 | value_scaling=True mate_defence=1 fifty_horizon=18 pawn_weight=1.075 reply_weight=0.5678 repeat_as_draw=False |
| 223 | t0585 | 0.500 | 0.566 | 1/1/1 | 0.98 | 7.90 | repeat_as_draw=False pawn_weight=0.5447 reply_weight=0.7776 quiesce_promotions=False value_scaling=True king_weight=0.05553 ahead_value=0.5 mate_attack=3 ahead_material=0.6085 |
| 224 | t0158 | 0.500 | 0.566 | 1/1/1 | 0.98 | 7.93 | safety_weight=4.612 quiesce_promotions=False fifty_horizon=26 reply_top=1 draw_value=0.2873 value_scaling=True |
| 225 | t0140 | 0.500 | 0.566 | 1/1/1 | 0.98 | 7.69 | reply_weight=0.2385 king_weight=0.04787 tactic_nodes=574931 pawn_convert=0.5937 simplify_weight=0.139 pawn_weight=1.421 scan_keep=10 ahead_material=0.6041 mate_attack=1 simplify_margin=2.765 |
| 226 | t0750 | 0.500 | 0.566 | 1/1/1 | 0.99 | 7.89 | simplify_margin=1.659 phase_values=1.000,2.069,2.139,3.515,9.612;1.000,2.069,2.139,3.515,9.612;1.000,2.069,2.139,3.515,9.612 pawn_weight=0.1229 draw_value=0.3818 value_scaling=True |
| 227 | t0982 | 0.500 | 0.566 | 1/1/1 | 0.99 | 8.33 | fifty_horizon=16 opening_weight=0.1794 quiesce_promotions=False simplify_weight=0.06271 ahead_value=0.555 ahead_draws=True |
| 228 | t0513 | 0.500 | 0.566 | 1/1/1 | 0.99 | 8.18 | value_scaling=True opening_weight=0.05179 safety_weight=4.902 ahead_draws=True tactic_nodes=556363 tactic_top=9 draw_value=0.2259 mate_defence=1 mate_attack=1 simplify_weight=0.1009 |
| 229 | t0811 | 0.500 | 0.566 | 1/1/1 | 0.99 | 8.05 | pawn_weight=0.2002 repeat_as_draw=False tactic_top=12 phase_values=1.000,2.899,2.001,5.870,7.355;1.000,2.899,2.001,5.870,7.355;1.000,2.899,2.001,5.870,7.355 tactic_rounds=2 |
| 230 | t0933 | 0.500 | 0.566 | 1/1/1 | 0.99 | 8.33 | fifty_horizon=0 pawn_convert=1 value_scaling=True mate_defence=1 simplify_weight=0.06122 tactic_rounds=1 opening_weight=0.1167 tactic_nodes=1444522 |
| 231 | t0915 | 0.500 | 0.566 | 0/3/0 | 1.00 | 8.37 | mate_defence=1 safety_weight=6 simplify_weight=0.04281 tactic_rounds=1 phase_values=1.000,2.951,1.997,3.832,12.331;1.000,2.951,1.997,3.832,12.331;1.000,2.951,1.997,3.832,12.331 draw_value=0.2 ahead_value=0.8433 repeat_as_draw=False quiesce_promotions=False tactic_nodes=1271286 |
| 232 | t0093 | 0.500 | 0.566 | 1/1/1 | 1.00 | 8.25 | opening_weight=0.4879 draw_value=0.6 pawn_weight=0.3911 mate_attack=1 tactic_nodes=310196 king_weight=0.05388 simplify_margin=1.932 safety_weight=5.295 reply_weight=0.09356 ahead_material=0.9671 |
| 233 | t0787 | 0.500 | 0.566 | 1/1/1 | 1.00 | 8.15 | value_scaling=True fifty_horizon=6 tactic_nodes=611874 reply_top=1 safety_weight=4.373 tactic_rounds=1 scan_keep=29 simplify_weight=0.05932 opening_weight=0.1117 |
| 234 | t0664 | 0.500 | 0.566 | 1/1/1 | 1.00 | 8.11 | tactic_nodes=648488 phase_values=1.000,2.818,2.943,3.946,10.647;1.000,2.818,2.943,3.946,10.647;1.000,2.818,2.943,3.946,10.647 pawn_weight=0.5435 king_weight=0.3799 draw_value=0.409 quiesce_promotions=False fifty_horizon=27 scan_keep=23 repeat_as_draw=False tactic_rounds=2 |
| 235 | t0566 | 0.333 | 0.533 | 0/2/1 | 0.25 | 2.01 | scan_depth=0 phase_values=1.000,2.704,2.359,3.347,8.162;1.000,2.704,2.359,3.347,8.162;1.000,2.704,2.359,3.347,8.162 tactic_depth=4 value_scaling=True scan_keep=22 simplify_weight=0.01498 |
| 236 | t0715 | 0.333 | 0.533 | 0/2/1 | 0.27 | 2.09 | draw_value=0.5379 scan_depth=1 value_scaling=True mate_attack=3 simplify_margin=0.3391 pawn_convert=1 scan_keep=13 tactic_depth=4 ahead_material=1 tactic_rounds=1 |
| 237 | t0357 | 0.333 | 0.533 | 0/2/1 | 0.27 | 2.19 | tactic_nodes=1500000 scan_depth=1 tactic_depth=4 opening_weight=0.5669 reply_weight=0.1683 draw_value=0.3237 |
| 238 | t0679 | 0.333 | 0.533 | 0/2/1 | 0.28 | 2.22 | tactic_nodes=1500000 quiesce_promotions=False phase_values=1.000,2.760,1.979,4.700,8.039;1.000,2.760,1.979,4.700,8.039;1.000,2.760,1.979,4.700,8.039 pawn_convert=0.1467 ahead_value=0.5482 mate_defence=1 reply_weight=0.4143 scan_depth=1 tactic_depth=4 |
| 239 | t0882 | 0.333 | 0.533 | 1/0/2 | 0.44 | 3.65 | tactic_depth=4 scan_depth=1 reply_top=3 fifty_horizon=28 pawn_convert=0.1259 pawn_weight=1.686 |
| 240 | t0009 | 0.333 | 0.533 | 0/2/1 | 0.46 | 5.00 | tactic_top=5 scan_keep=0 simplify_weight=0.3804 pawn_convert=1 mate_attack=1 tactic_rounds=1 scan_depth=0 |
| 241 | t0519 | 0.333 | 0.533 | 0/2/1 | 0.48 | 3.88 | tactic_depth=5 tactic_rounds=4 draw_value=0.2154 scan_depth=1 reply_weight=0.1972 check_extension=1 safety_weight=3.09 pawn_convert=0.3012 ahead_value=0.9 |
| 242 | t0133 | 0.333 | 0.533 | 1/0/2 | 0.56 | 4.55 | pawn_weight=0.1256 safety_weight=0.7832 quiesce_promotions=False reply_top=3 tactic_depth=4 tactic_top=9 repeat_as_draw=False pawn_convert=0.4052 scan_depth=2 |
| 243 | t0731 | 0.333 | 0.533 | 1/0/2 | 0.59 | 4.83 | reply_weight=0.1607 safety_weight=3.594 mate_defence=1 tactic_depth=4 king_weight=0.02559 ahead_value=0.5868 tactic_rounds=2 |
| 244 | t0690 | 0.333 | 0.533 | 1/0/2 | 0.60 | 4.74 | tactic_nodes=178323 safety_weight=5.256 scan_keep=6 exchange_depth=3 tactic_depth=4 |
| 245 | t0163 | 0.333 | 0.533 | 1/0/2 | 0.61 | 4.85 | simplify_margin=1.237 fifty_horizon=24 exchange_depth=6 ahead_material=0.5616 tactic_depth=4 |
| 246 | t0602 | 0.333 | 0.533 | 1/0/2 | 0.61 | 5.04 | king_weight=0.3098 simplify_weight=0.06127 ahead_material=0.6603 tactic_depth=4 safety_weight=6 ahead_draws=True mate_attack=3 reply_weight=0.5533 simplify_margin=1.772 tactic_nodes=555922 |
| 247 | t0907 | 0.333 | 0.533 | 0/2/1 | 0.63 | 5.16 | tactic_nodes=883759 pawn_weight=1.653 phase_values=1.000,2.255,3.499,3.046,6.069;1.000,2.255,3.499,3.046,6.069;1.000,2.255,3.499,3.046,6.069 mate_defence=3 tactic_depth=4 tactic_rounds=2 scan_keep=2 check_extension=1 |
| 248 | t0816 | 0.333 | 0.533 | 1/0/2 | 0.64 | 5.01 | simplify_weight=0.1034 tactic_depth=4 safety_weight=5.972 tactic_top=13 pawn_weight=1.422 |
| 249 | t1022 | 0.333 | 0.533 | 0/2/1 | 0.64 | 4.34 | ahead_material=0.1785 mate_attack=3 pawn_convert=1 scan_keep=4 tactic_depth=5 opening_weight=0.1336 pawn_weight=1.552 mate_defence=3 |
| 250 | t0330 | 0.333 | 0.533 | 1/0/2 | 0.64 | 5.19 | reply_top=1 ahead_material=0.06976 ahead_value=0.8911 opening_weight=0.2018 simplify_margin=1.385 ahead_draws=True quiesce_promotions=False king_weight=0.4902 mate_attack=3 tactic_depth=4 |
| 251 | t0814 | 0.333 | 0.533 | 1/0/2 | 0.65 | 5.16 | tactic_rounds=1 pawn_convert=0.5039 draw_value=0.2851 scan_depth=0 fifty_horizon=16 quiesce_promotions=False exchange_depth=6 tactic_nodes=854708 mate_defence=3 repeat_as_draw=False |
| 252 | t0838 | 0.333 | 0.533 | 1/0/2 | 0.65 | 5.31 | reply_top=1 tactic_depth=4 pawn_weight=1.803 king_weight=0.4154 safety_weight=1.365 draw_value=0.6 pawn_convert=0.163 |
| 253 | t0660 | 0.333 | 0.533 | 0/2/1 | 0.65 | 5.25 | simplify_weight=0.2089 quiesce_promotions=False phase_values=1.000,2.145,2.412,2.831,6.664;1.000,2.145,2.412,2.831,6.664;1.000,2.145,2.412,2.831,6.664 repeat_as_draw=False scan_depth=1 ahead_value=0.9 opening_weight=0.5364 |
| 254 | t0432 | 0.333 | 0.533 | 1/0/2 | 0.65 | 5.51 | mate_attack=1 tactic_rounds=2 simplify_margin=1.102 fifty_horizon=29 ahead_draws=True tactic_depth=4 |
| 255 | t0492 | 0.333 | 0.533 | 1/0/2 | 0.67 | 5.22 | draw_value=0.3529 simplify_margin=2.386 simplify_weight=0.08264 scan_keep=6 reply_weight=0.7988 scan_depth=2 pawn_weight=1.973 quiesce_promotions=False |
| 256 | t0800 | 0.333 | 0.533 | 1/0/2 | 0.67 | 5.20 | tactic_depth=4 scan_keep=27 safety_weight=3.651 phase_values=1.000,2.947,2.561,3.100,7.952;1.000,2.947,2.561,3.100,7.952;1.000,2.947,2.561,3.100,7.952 king_weight=0.02318 quiesce_promotions=False |
| - | base | 0.667 | 0.533 | 2/0/1 | 1.00 | 8.12 | (base: --rapido + m039) |

## Fase 2: 256 x bo10

2570 jogos, 0.44 s/jogo

| # | id | score | ±95% | V/E/D | tempo vs base | ms/lance (prova) | mutações |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | t0205 | 0.850 | 0.221 | 8/1/1 | 0.62 | 5.09 | ahead_draws=True tactic_depth=4 draw_value=0.4334 pawn_convert=0.1947 mate_attack=3 simplify_weight=0.2647 |
| 2 | t0939 | 0.850 | 0.221 | 7/3/0 | 0.97 | 8.04 | fifty_horizon=23 ahead_material=0.02772 exchange_depth=7 simplify_weight=0.04869 pawn_weight=1.693 ahead_value=0.7274 reply_weight=0.2487 mate_defence=3 opening_weight=0.1192 repeat_as_draw=False |
| 3 | t0145 | 0.800 | 0.248 | 6/4/0 | 0.99 | 7.93 | tactic_rounds=4 king_weight=0.3966 ahead_draws=True mate_attack=1 fifty_horizon=19 reply_top=1 pawn_convert=1 |
| 4 | t0007 | 0.800 | 0.248 | 7/2/1 | 0.95 | 9.97 | mate_attack=1 draw_value=0.4582 exchange_depth=4 opening_weight=0.05612 ahead_draws=True pawn_weight=0.3317 safety_weight=5.9 check_extension=1 tactic_top=8 |
| 5 | t0396 | 0.750 | 0.268 | 7/1/2 | 0.96 | 8.27 | mate_attack=3 simplify_margin=2.002 check_extension=1 phase_values=1.000,1.762,2.083,4.964,11.600;1.000,1.762,2.083,4.964,11.600;1.000,1.762,2.083,4.964,11.600 scan_depth=2 king_weight=0.4399 pawn_weight=0.594 pawn_convert=0.7447 fifty_horizon=14 |
| 6 | t0083 | 0.750 | 0.268 | 6/3/1 | 0.99 | 8.19 | tactic_rounds=2 pawn_weight=0.6631 scan_keep=25 draw_value=0.6 reply_top=1 |
| 7 | t0467 | 0.750 | 0.268 | 7/1/2 | 0.95 | 7.68 | scan_keep=12 value_scaling=True reply_weight=0.2404 fifty_horizon=28 safety_weight=3.416 |
| 8 | t0730 | 0.700 | 0.284 | 5/4/1 | 0.87 | 7.02 | exchange_depth=4 tactic_top=7 pawn_weight=0.5369 scan_keep=16 safety_weight=3.208 |
| 9 | t0650 | 0.700 | 0.284 | 5/4/1 | 0.90 | 7.26 | exchange_depth=7 mate_attack=3 simplify_margin=0.1616 safety_weight=3.069 ahead_material=0.5766 pawn_convert=0.6579 mate_defence=1 simplify_weight=0.4843 fifty_horizon=19 |
| 10 | t0116 | 0.700 | 0.284 | 6/2/2 | 0.94 | 7.76 | tactic_depth=5 pawn_weight=0.5818 ahead_value=0.7378 scan_keep=12 reply_top=4 safety_weight=6 |
| 11 | t0051 | 0.700 | 0.284 | 6/2/2 | 0.97 | 7.83 | scan_depth=2 tactic_top=12 scan_keep=0 tactic_rounds=2 check_extension=1 pawn_convert=0.4096 ahead_value=0.8186 |
| 12 | t0558 | 0.700 | 0.284 | 6/2/2 | 0.98 | 7.93 | tactic_top=7 pawn_convert=0.6669 safety_weight=2.558 mate_attack=1 value_scaling=True tactic_nodes=1500000 ahead_draws=True fifty_horizon=8 king_weight=0.01457 draw_value=0.2301 |
| 13 | t0111 | 0.700 | 0.284 | 6/2/2 | 0.96 | 7.62 | mate_attack=1 ahead_material=0.9927 simplify_margin=0.9805 pawn_convert=1 reply_top=1 exchange_depth=4 |
| 14 | t0158 | 0.700 | 0.284 | 5/4/1 | 0.98 | 7.93 | safety_weight=4.612 quiesce_promotions=False fifty_horizon=26 reply_top=1 draw_value=0.2873 value_scaling=True |
| 15 | t0605 | 0.650 | 0.296 | 6/1/3 | 1.00 | 8.01 | opening_weight=0.08134 safety_weight=6 tactic_rounds=4 fifty_horizon=4 ahead_draws=True value_scaling=True |
| 16 | t0289 | 0.650 | 0.296 | 5/3/2 | 0.68 | 5.49 | repeat_as_draw=False mate_attack=3 tactic_rounds=2 ahead_material=0.4954 scan_depth=1 opening_weight=0.08165 safety_weight=3.095 tactic_top=12 draw_value=0.6 pawn_convert=0.8554 |
| 17 | t0308 | 0.650 | 0.296 | 4/5/1 | 0.73 | 5.89 | scan_depth=2 reply_top=1 tactic_nodes=1500000 reply_weight=0.3608 scan_keep=13 repeat_as_draw=False safety_weight=1.466 |
| 18 | t0794 | 0.650 | 0.296 | 5/3/2 | 0.74 | 5.97 | simplify_margin=1.328 tactic_top=7 ahead_material=1 safety_weight=6 opening_weight=0.02585 simplify_weight=0.3971 mate_attack=1 quiesce_promotions=False scan_depth=2 |
| 19 | t0008 | 0.650 | 0.296 | 6/1/3 | 0.83 | 8.60 | king_weight=0.4412 tactic_top=4 opening_weight=0.03536 reply_top=1 phase_values=1.000,1.741,2.336,4.061,11.839;1.000,1.741,2.336,4.061,11.839;1.000,1.741,2.336,4.061,11.839 |
| 20 | t0348 | 0.650 | 0.296 | 5/3/2 | 0.99 | 7.82 | fifty_horizon=19 ahead_value=0.9 quiesce_promotions=False pawn_convert=1 tactic_nodes=292624 opening_weight=0.1189 tactic_rounds=1 |
| 21 | t0591 | 0.650 | 0.296 | 5/3/2 | 0.99 | 8.10 | reply_top=2 reply_weight=0.3301 tactic_rounds=1 fifty_horizon=14 check_extension=1 pawn_weight=0.1856 ahead_value=0.8051 simplify_margin=0.5324 repeat_as_draw=False scan_depth=1 |
| 22 | t0286 | 0.650 | 0.296 | 6/1/3 | 0.76 | 6.11 | reply_weight=0.4277 tactic_depth=5 opening_weight=0.1101 value_scaling=True simplify_margin=0.5323 |
| 23 | t0652 | 0.650 | 0.296 | 5/3/2 | 0.84 | 6.80 | mate_defence=1 pawn_weight=0.2647 check_extension=1 ahead_draws=True tactic_top=6 mate_attack=3 scan_depth=0 tactic_nodes=815246 tactic_rounds=4 |
| 24 | t0556 | 0.650 | 0.296 | 6/1/3 | 0.87 | 7.08 | tactic_nodes=1500000 exchange_depth=6 fifty_horizon=0 safety_weight=6 reply_top=1 ahead_value=0.5 |
| 25 | t0290 | 0.650 | 0.296 | 6/1/3 | 0.94 | 7.71 | pawn_convert=0.3174 ahead_value=0.6994 simplify_margin=0.1272 reply_weight=0.8997 pawn_weight=1.396 quiesce_promotions=False |
| 26 | t0623 | 0.650 | 0.296 | 5/3/2 | 0.47 | 3.81 | simplify_margin=2.196 ahead_material=0.3531 safety_weight=4.229 check_extension=1 quiesce_promotions=False tactic_depth=4 scan_depth=2 mate_defence=3 |
| 27 | t0170 | 0.650 | 0.296 | 5/3/2 | 0.54 | 4.42 | fifty_horizon=22 repeat_as_draw=False value_scaling=True scan_depth=2 check_extension=1 scan_keep=9 mate_attack=1 tactic_rounds=4 tactic_depth=5 |
| 28 | t0551 | 0.650 | 0.296 | 5/3/2 | 0.87 | 7.10 | quiesce_promotions=False tactic_nodes=718573 reply_top=2 scan_keep=23 simplify_margin=0.8861 tactic_depth=5 exchange_depth=7 tactic_rounds=4 king_weight=0.243 |
| 29 | t0162 | 0.650 | 0.296 | 5/3/2 | 0.88 | 7.05 | tactic_top=7 mate_attack=1 simplify_margin=1.219 scan_depth=2 check_extension=1 ahead_material=0.9318 draw_value=0.3616 |
| 30 | t0118 | 0.650 | 0.296 | 6/1/3 | 0.90 | 7.40 | pawn_weight=0.3154 ahead_material=0.8631 reply_top=1 simplify_margin=2.036 simplify_weight=0.01642 scan_keep=6 ahead_value=0.9 |
| 31 | t0789 | 0.650 | 0.296 | 5/3/2 | 0.91 | 7.27 | simplify_margin=1.861 tactic_top=9 scan_keep=0 mate_attack=1 phase_values=1.000,2.687,2.132,4.302,7.032;1.000,2.687,2.132,4.302,7.032;1.000,2.687,2.132,4.302,7.032 pawn_convert=0.9849 exchange_depth=4 pawn_weight=1.728 tactic_rounds=4 |
| 32 | t0003 | 0.650 | 0.296 | 5/3/2 | 0.91 | 9.58 | ahead_draws=True scan_keep=12 mate_attack=3 repeat_as_draw=False reply_top=3 simplify_weight=0.262 |
| 33 | t0792 | 0.650 | 0.296 | 4/5/1 | 0.96 | 7.82 | mate_attack=3 safety_weight=5.137 scan_depth=2 tactic_nodes=155645 exchange_depth=3 phase_values=1.000,2.630,3.354,4.472,6.500;1.000,2.630,3.354,4.472,6.500;1.000,2.630,3.354,4.472,6.500 pawn_convert=0.8547 ahead_material=0.2868 tactic_rounds=2 |
| 34 | t0709 | 0.650 | 0.296 | 6/1/3 | 0.96 | 7.78 | reply_weight=0.1691 pawn_weight=1.69 phase_values=1.000,2.465,3.806,3.201,9.483;1.000,2.465,3.806,3.201,9.483;1.000,2.465,3.806,3.201,9.483 opening_weight=0.03089 exchange_depth=4 pawn_convert=0.9575 quiesce_promotions=False |
| 35 | t0140 | 0.650 | 0.296 | 5/3/2 | 0.98 | 7.69 | reply_weight=0.2385 king_weight=0.04787 tactic_nodes=574931 pawn_convert=0.5937 simplify_weight=0.139 pawn_weight=1.421 scan_keep=10 ahead_material=0.6041 mate_attack=1 simplify_margin=2.765 |
| 36 | t0933 | 0.650 | 0.296 | 5/3/2 | 0.99 | 8.33 | fifty_horizon=0 pawn_convert=1 value_scaling=True mate_defence=1 simplify_weight=0.06122 tactic_rounds=1 opening_weight=0.1167 tactic_nodes=1444522 |
| 37 | t0271 | 0.600 | 0.304 | 3/6/1 | 0.99 | 8.11 | mate_attack=3 tactic_rounds=1 pawn_convert=0.2082 exchange_depth=6 pawn_weight=0.1924 repeat_as_draw=False mate_defence=1 value_scaling=True scan_keep=8 check_extension=1 |
| 38 | t0757 | 0.600 | 0.304 | 5/2/3 | 0.94 | 7.58 | king_weight=0.109 tactic_nodes=148783 reply_weight=0.9543 ahead_draws=True exchange_depth=4 ahead_value=0.8044 mate_attack=1 pawn_weight=1.759 value_scaling=True fifty_horizon=24 |
| 39 | t0825 | 0.600 | 0.304 | 6/0/4 | 0.97 | 7.64 | draw_value=0.385 pawn_convert=0.2145 value_scaling=True opening_weight=0.1259 exchange_depth=6 tactic_rounds=4 mate_defence=3 |
| 40 | t0193 | 0.600 | 0.304 | 4/4/2 | 0.99 | 7.93 | scan_keep=7 check_extension=1 draw_value=0.2742 pawn_convert=1 exchange_depth=6 simplify_weight=0.1221 reply_weight=0.1775 tactic_nodes=1500000 |
| 41 | t0385 | 0.600 | 0.304 | 4/4/2 | 0.99 | 8.11 | tactic_nodes=1500000 ahead_material=0.7913 scan_keep=22 simplify_margin=2.015 opening_weight=0.4987 repeat_as_draw=False |
| 42 | t0718 | 0.600 | 0.304 | 4/4/2 | 0.68 | 5.40 | reply_top=1 tactic_top=4 mate_defence=1 safety_weight=1.557 exchange_depth=7 tactic_depth=5 ahead_value=0.9 |
| 43 | t0302 | 0.600 | 0.304 | 5/2/3 | 0.75 | 6.18 | pawn_convert=0.3461 ahead_draws=True quiesce_promotions=False scan_depth=2 value_scaling=True |
| 44 | t0291 | 0.600 | 0.304 | 4/4/2 | 0.76 | 5.80 | scan_keep=23 scan_depth=2 tactic_nodes=956567 pawn_weight=0.9918 ahead_draws=True simplify_weight=0.4861 safety_weight=3.714 simplify_margin=0.6293 draw_value=0.6 exchange_depth=7 |
| 45 | t0720 | 0.600 | 0.304 | 5/2/3 | 0.87 | 7.25 | scan_keep=17 check_extension=1 scan_depth=2 fifty_horizon=9 king_weight=0.0405 ahead_material=0.253 tactic_rounds=4 tactic_nodes=438850 |
| 46 | t0886 | 0.600 | 0.304 | 5/2/3 | 0.89 | 7.72 | mate_defence=1 scan_keep=18 safety_weight=4.339 pawn_weight=0.762 tactic_rounds=2 phase_values=1.000,2.348,2.268,4.266,11.714;1.000,2.348,2.268,4.266,11.714;1.000,2.348,2.268,4.266,11.714 tactic_top=5 reply_weight=0.9363 repeat_as_draw=False simplify_margin=0.04824 |
| 47 | t0504 | 0.600 | 0.304 | 4/4/2 | 0.92 | 7.52 | quiesce_promotions=False reply_weight=0.07543 mate_attack=3 fifty_horizon=0 exchange_depth=7 tactic_rounds=4 king_weight=0.09215 |
| 48 | t0188 | 0.600 | 0.304 | 5/2/3 | 0.97 | 7.69 | simplify_margin=1.726 tactic_top=6 pawn_convert=0.6717 king_weight=0.2728 ahead_value=0.5 |
| 49 | t0124 | 0.600 | 0.304 | 5/2/3 | 0.74 | 6.08 | simplify_margin=0.154 tactic_top=9 tactic_rounds=2 pawn_convert=0.9126 reply_weight=0.1972 exchange_depth=4 scan_depth=0 reply_top=2 |
| 50 | t0719 | 0.600 | 0.304 | 5/2/3 | 0.80 | 6.24 | quiesce_promotions=False king_weight=0.3103 opening_weight=0.5566 tactic_rounds=1 tactic_top=11 scan_depth=2 pawn_convert=0.7823 ahead_value=0.6333 |
| 51 | t0035 | 0.600 | 0.304 | 3/6/1 | 0.93 | 8.03 | quiesce_promotions=False tactic_rounds=4 pawn_convert=0.479 scan_keep=15 ahead_material=0.9906 reply_top=1 |
| 52 | t0198 | 0.600 | 0.304 | 4/4/2 | 0.93 | 7.55 | exchange_depth=6 opening_weight=0.1414 pawn_weight=0.2265 fifty_horizon=15 ahead_value=0.6837 tactic_rounds=2 check_extension=1 mate_attack=1 simplify_margin=0.4654 tactic_depth=5 |
| 53 | t0065 | 0.600 | 0.304 | 5/2/3 | 0.98 | 7.96 | king_weight=0.06483 safety_weight=3.625 opening_weight=0.03764 reply_top=1 ahead_material=0.2336 |
| 54 | t0982 | 0.600 | 0.304 | 4/4/2 | 0.99 | 8.33 | fifty_horizon=16 opening_weight=0.1794 quiesce_promotions=False simplify_weight=0.06271 ahead_value=0.555 ahead_draws=True |
| 55 | t0787 | 0.600 | 0.304 | 6/0/4 | 1.00 | 8.15 | value_scaling=True fifty_horizon=6 tactic_nodes=611874 reply_top=1 safety_weight=4.373 tactic_rounds=1 scan_keep=29 simplify_weight=0.05932 opening_weight=0.1117 |
| 56 | t0560 | 0.550 | 0.308 | 4/3/3 | 0.87 | 7.09 | tactic_nodes=268553 exchange_depth=7 king_weight=0.3674 ahead_material=0.5233 draw_value=0.2 mate_defence=1 |
| 57 | t0721 | 0.550 | 0.308 | 3/5/2 | 0.98 | 7.78 | repeat_as_draw=False tactic_nodes=1018048 opening_weight=0.2114 pawn_convert=1 ahead_draws=True tactic_top=8 quiesce_promotions=False reply_weight=0.9429 |
| 58 | t0767 | 0.550 | 0.308 | 5/1/4 | 0.71 | 5.66 | quiesce_promotions=False tactic_top=6 mate_attack=3 fifty_horizon=9 tactic_depth=4 simplify_weight=0.1329 mate_defence=3 safety_weight=3.439 |
| 59 | t0366 | 0.550 | 0.308 | 4/3/3 | 0.81 | 6.32 | ahead_material=1 reply_weight=0.01926 reply_top=2 tactic_rounds=2 scan_depth=1 |
| 60 | t0670 | 0.550 | 0.308 | 4/3/3 | 0.87 | 6.97 | tactic_top=6 scan_depth=1 repeat_as_draw=False value_scaling=True reply_weight=0.09228 quiesce_promotions=False fifty_horizon=9 check_extension=1 phase_values=1.000,1.595,1.931,4.039,11.525;1.000,1.595,1.931,4.039,11.525;1.000,1.595,1.931,4.039,11.525 |
| 61 | t0987 | 0.550 | 0.308 | 4/3/3 | 0.88 | 7.37 | fifty_horizon=1 ahead_material=1 tactic_top=7 draw_value=0.4843 simplify_margin=1.72 exchange_depth=7 pawn_convert=0.9093 reply_weight=0.6055 |
| 62 | t0534 | 0.550 | 0.308 | 4/3/3 | 0.96 | 7.47 | safety_weight=2.652 mate_attack=3 opening_weight=0.2858 tactic_rounds=1 fifty_horizon=2 exchange_depth=7 ahead_value=0.5255 |
| 63 | t0828 | 0.550 | 0.308 | 3/5/2 | 0.99 | 8.00 | mate_defence=1 pawn_weight=0.1945 opening_weight=0.05605 simplify_margin=0.2566 tactic_rounds=1 king_weight=0.4266 quiesce_promotions=False tactic_nodes=279065 reply_weight=0.05763 repeat_as_draw=False |
| 64 | t0137 | 0.550 | 0.308 | 5/1/4 | 1.00 | 7.99 | tactic_nodes=141387 reply_weight=0.965 pawn_convert=0.3019 mate_attack=1 quiesce_promotions=False |
| - | base | 0.750 | 0.268 | 7/1/2 | 1.00 | 8.12 | (base: --rapido + m039) |

## Fase 3: 64 x bo30

1950 jogos, 0.47 s/jogo

| # | id | score | ±95% | V/E/D | tempo vs base | ms/lance (prova) | mutações |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | t0933 | 0.683 | 0.166 | 17/7/6 | 0.99 | 8.33 | fifty_horizon=0 pawn_convert=1 value_scaling=True mate_defence=1 simplify_weight=0.06122 tactic_rounds=1 opening_weight=0.1167 tactic_nodes=1444522 |
| 2 | t0719 | 0.683 | 0.166 | 16/9/5 | 0.80 | 6.24 | quiesce_promotions=False king_weight=0.3103 opening_weight=0.5566 tactic_rounds=1 tactic_top=11 scan_depth=2 pawn_convert=0.7823 ahead_value=0.6333 |
| 3 | t0111 | 0.650 | 0.171 | 14/11/5 | 0.96 | 7.62 | mate_attack=1 ahead_material=0.9927 simplify_margin=0.9805 pawn_convert=1 reply_top=1 exchange_depth=4 |
| 4 | t0560 | 0.617 | 0.174 | 15/7/8 | 0.87 | 7.09 | tactic_nodes=268553 exchange_depth=7 king_weight=0.3674 ahead_material=0.5233 draw_value=0.2 mate_defence=1 |
| 5 | t0551 | 0.617 | 0.174 | 15/7/8 | 0.87 | 7.10 | quiesce_promotions=False tactic_nodes=718573 reply_top=2 scan_keep=23 simplify_margin=0.8861 tactic_depth=5 exchange_depth=7 tactic_rounds=4 king_weight=0.243 |
| 6 | t0987 | 0.617 | 0.174 | 15/7/8 | 0.88 | 7.37 | fifty_horizon=1 ahead_material=1 tactic_top=7 draw_value=0.4843 simplify_margin=1.72 exchange_depth=7 pawn_convert=0.9093 reply_weight=0.6055 |
| 7 | t0939 | 0.600 | 0.175 | 11/14/5 | 0.97 | 8.04 | fifty_horizon=23 ahead_material=0.02772 exchange_depth=7 simplify_weight=0.04869 pawn_weight=1.693 ahead_value=0.7274 reply_weight=0.2487 mate_defence=3 opening_weight=0.1192 repeat_as_draw=False |
| 8 | t0137 | 0.600 | 0.175 | 14/8/8 | 1.00 | 7.99 | tactic_nodes=141387 reply_weight=0.965 pawn_convert=0.3019 mate_attack=1 quiesce_promotions=False |
| 9 | t0348 | 0.583 | 0.176 | 14/7/9 | 0.99 | 7.82 | fifty_horizon=19 ahead_value=0.9 quiesce_promotions=False pawn_convert=1 tactic_nodes=292624 opening_weight=0.1189 tactic_rounds=1 |
| 10 | t0825 | 0.583 | 0.176 | 11/13/6 | 0.97 | 7.64 | draw_value=0.385 pawn_convert=0.2145 value_scaling=True opening_weight=0.1259 exchange_depth=6 tactic_rounds=4 mate_defence=3 |
| 11 | t0158 | 0.583 | 0.176 | 13/9/8 | 0.98 | 7.93 | safety_weight=4.612 quiesce_promotions=False fifty_horizon=26 reply_top=1 draw_value=0.2873 value_scaling=True |
| 12 | t0605 | 0.567 | 0.177 | 12/10/8 | 1.00 | 8.01 | opening_weight=0.08134 safety_weight=6 tactic_rounds=4 fifty_horizon=4 ahead_draws=True value_scaling=True |
| 13 | t0467 | 0.567 | 0.177 | 13/8/9 | 0.95 | 7.68 | scan_keep=12 value_scaling=True reply_weight=0.2404 fifty_horizon=28 safety_weight=3.416 |
| 14 | t0051 | 0.567 | 0.177 | 15/4/11 | 0.97 | 7.83 | scan_depth=2 tactic_top=12 scan_keep=0 tactic_rounds=2 check_extension=1 pawn_convert=0.4096 ahead_value=0.8186 |
| 15 | t0193 | 0.567 | 0.177 | 14/6/10 | 0.99 | 7.93 | scan_keep=7 check_extension=1 draw_value=0.2742 pawn_convert=1 exchange_depth=6 simplify_weight=0.1221 reply_weight=0.1775 tactic_nodes=1500000 |
| 16 | t0162 | 0.567 | 0.177 | 14/6/10 | 0.88 | 7.05 | tactic_top=7 mate_attack=1 simplify_margin=1.219 scan_depth=2 check_extension=1 ahead_material=0.9318 draw_value=0.3616 |
| - | base | 0.617 | 0.174 | 16/5/9 | 1.00 | 8.12 | (base: --rapido + m039) |

## Fase 4: 16 x bo100

1700 jogos, 0.49 s/jogo

| # | id | score | ±95% | V/E/D | tempo vs base | ms/lance (prova) | mutações |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | t0467 | 0.620 | 0.095 | 47/30/23 | 0.95 | 7.68 | scan_keep=12 value_scaling=True reply_weight=0.2404 fifty_horizon=28 safety_weight=3.416 |
| 2 | t0987 | 0.615 | 0.095 | 51/21/28 | 0.88 | 7.37 | fifty_horizon=1 ahead_material=1 tactic_top=7 draw_value=0.4843 simplify_margin=1.72 exchange_depth=7 pawn_convert=0.9093 reply_weight=0.6055 |
| - | base | 0.600 | 0.096 | 47/26/27 | 1.00 | 8.12 | (base: --rapido + m039) |

## Fase 5: 2 x bo500

1500 jogos, 0.50 s/jogo

| # | id | score | ±95% | V/E/D | tempo vs base | ms/lance (prova) | mutações |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | t0467 | 0.596 | 0.043 | 233/130/137 | 0.95 | 7.68 | scan_keep=12 value_scaling=True reply_weight=0.2404 fifty_horizon=28 safety_weight=3.416 |
| 2 | t0987 | 0.516 | 0.044 | 177/162/161 | 0.88 | 7.37 | fifty_horizon=1 ahead_material=1 tactic_top=7 draw_value=0.4843 simplify_margin=1.72 exchange_depth=7 pawn_convert=0.9093 reply_weight=0.6055 |
| - | base | 0.570 | 0.043 | 210/150/140 | 1.00 | 8.12 | (base: --rapido + m039) |

## Conclusão

Prova de tempo: 470 de 1024 mutantes não mais lentos que a base (mediana 1,02x). 9133 jogos, 0 erros.
Final (500 jogos cada, intercalados com a base): t0467 0,596, base 0,570, t0987 0,516 (±0,043).
t0467 é ~5% mais rápido (6,4 vs 6,7 ms por lance em jogo) e +0,026 (dentro da margem: não provado melhor).
t0987 é mais rápido (6,2 ms) mas pior (-0,054).
