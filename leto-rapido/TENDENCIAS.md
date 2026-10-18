
### Torneio com prova de tempo (fase 1, bo3; prova de tempo de todos os 1024): 470 mutantes, score médio 0.421

| mutação | direção | n | efeito no score | ±1σ | mortos na prova | tempo mediano |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| check_extension | ↑ | 78 | +0.072 | 0.029 | 72% | 1.28 |
| pawn_convert | ↑ | 60 | +0.070 | 0.031 | 57% | 1.04 |
| opening_weight | ↓ | 43 | +0.055 | 0.036 | 49% | 1.00 |
| exchange_depth | ↑ | 84 | +0.053 | 0.029 | 44% | 0.98 |
| pawn_weight | ↓ | 51 | +0.049 | 0.039 | 55% | 1.01 |
| simplify_margin | ↓ | 42 | +0.043 | 0.038 | 53% | 1.01 |
| fifty_horizon | ↓ | 44 | +0.029 | 0.045 | 56% | 1.02 |
| ahead_material | ↑ | 68 | +0.027 | 0.030 | 53% | 1.02 |
| pawn_weight | ↑ | 74 | +0.027 | 0.027 | 55% | 1.01 |
| reply_weight | ↑ | 62 | +0.025 | 0.033 | 58% | 1.02 |
| opening_weight | ↑ | 84 | +0.023 | 0.029 | 57% | 1.03 |
| ahead_value | ↓ | 40 | +0.020 | 0.038 | 59% | 1.04 |
| tactic_top | ↓ | 87 | +0.017 | 0.027 | 43% | 0.98 |
| mate_attack | ↓ | 64 | +0.016 | 0.031 | 56% | 1.02 |
| scan_keep | ↑ | 32 | +0.016 | 0.036 | 63% | 1.05 |
| tactic_top | ↑ | 40 | +0.016 | 0.035 | 63% | 1.11 |
| reply_top | ↑ | 91 | +0.015 | 0.027 | 68% | 1.11 |
| fifty_horizon | ↑ | 96 | +0.014 | 0.026 | 54% | 1.02 |
| scan_depth | ↓ | 163 | +0.011 | 0.020 | 15% | 0.78 |
| tactic_nodes | ↑ | 74 | +0.009 | 0.029 | 52% | 1.02 |
| quiesce_promotions | ↓ | 118 | +0.008 | 0.023 | 54% | 1.01 |
| ahead_value | ↑ | 75 | +0.008 | 0.028 | 52% | 1.00 |
| safety_weight | ↑ | 43 | +0.001 | 0.040 | 61% | 1.03 |
| pawn_convert | ↓ | 65 | -0.001 | 0.033 | 52% | 1.01 |
| simplify_margin | ↑ | 91 | -0.002 | 0.028 | 51% | 1.01 |
| tactic_rounds | ↑ | 35 | -0.002 | 0.046 | 62% | 1.04 |
| repeat_as_draw | ↓ | 120 | -0.003 | 0.024 | 55% | 1.02 |
| mate_defence | ↓ | 61 | -0.003 | 0.036 | 55% | 1.02 |
| draw_value | ↓ | 83 | -0.004 | 0.027 | 52% | 1.01 |
| ahead_draws | ↑ | 125 | -0.004 | 0.021 | 53% | 1.01 |
| safety_weight | ↓ | 92 | -0.006 | 0.026 | 48% | 0.99 |
| mate_attack | ↑ | 71 | -0.008 | 0.029 | 49% | 0.99 |
| value_scaling | ↑ | 123 | -0.009 | 0.024 | 54% | 1.02 |
| mate_defence | ↑ | 47 | -0.013 | 0.034 | 65% | 1.06 |
| tactic_rounds | ↓ | 88 | -0.016 | 0.025 | 48% | 1.00 |
| simplify_weight | ↓ | 28 | -0.017 | 0.048 | 59% | 1.06 |
| draw_value | ↑ | 39 | -0.024 | 0.040 | 57% | 1.04 |
| simplify_weight | ↑ | 90 | -0.027 | 0.028 | 57% | 1.02 |
| ahead_material | ↓ | 66 | -0.032 | 0.032 | 53% | 1.01 |
| tactic_nodes | ↓ | 59 | -0.037 | 0.036 | 57% | 1.02 |
| king_weight | ↑ | 86 | -0.041 | 0.029 | 55% | 1.02 |
| king_weight | ↓ | 29 | -0.042 | 0.043 | 70% | 1.12 |
| reply_weight | ↓ | 70 | -0.043 | 0.030 | 42% | 0.97 |
| piece_values | ↓ | 58 | -0.051 | 0.029 | 58% | 1.03 |
| scan_keep | ↓ | 106 | -0.064 | 0.026 | 48% | 0.99 |
| exchange_depth | ↓ | 59 | -0.065 | 0.032 | 58% | 1.09 |
| tactic_depth | ↓ | 158 | -0.067 | 0.019 | 15% | 0.75 |
| piece_values | ↑ | 49 | -0.068 | 0.032 | 58% | 1.04 |
| scan_depth | ↑ | 0 | +nan | nan | 100% | 1.64 |
| tactic_depth | ↑ | 0 | +nan | nan | 100% | 1.83 |

### Mega torneio (só julgamento, fase 1, bo20): 256 mutantes, score médio 0.518

| mutação | direção | n | efeito no score | ±1σ |
| --- | :---: | ---: | ---: | ---: |
| opening_weight | ↓ | 48 | +0.058 | 0.019 |
| safety_weight | ↑ | 73 | +0.058 | 0.013 |
| pawn_convert | ↓ | 55 | +0.027 | 0.019 |
| ahead_value | ↓ | 40 | +0.024 | 0.022 |
| reply_weight | ↑ | 67 | +0.020 | 0.020 |
| ahead_material | ↓ | 56 | +0.019 | 0.015 |
| ahead_value | ↑ | 67 | +0.018 | 0.016 |
| king_weight | ↓ | 39 | +0.015 | 0.018 |
| pawn_weight | ↓ | 46 | +0.013 | 0.017 |
| ahead_draws | ↓ | 122 | +0.005 | 0.012 |
| draw_value | ↓ | 67 | +0.001 | 0.020 |
| reply_weight | ↓ | 48 | +0.001 | 0.021 |
| simplify_weight | ↑ | 93 | +0.000 | 0.014 |
| fifty_horizon | ↓ | 34 | +0.000 | 0.027 |
| king_weight | ↑ | 84 | -0.003 | 0.017 |
| ahead_material | ↑ | 58 | -0.004 | 0.021 |
| pawn_convert | ↑ | 62 | -0.006 | 0.019 |
| draw_value | ↑ | 54 | -0.007 | 0.019 |
| simplify_margin | ↓ | 42 | -0.012 | 0.025 |
| fifty_horizon | ↑ | 78 | -0.014 | 0.017 |
| piece_values | ↑ | 67 | -0.015 | 0.015 |
| repeat_as_draw | ↓ | 119 | -0.016 | 0.012 |
| value_scaling | ↓ | 111 | -0.016 | 0.014 |
| simplify_margin | ↑ | 78 | -0.021 | 0.016 |
| simplify_weight | ↓ | 35 | -0.025 | 0.024 |
| pawn_weight | ↑ | 76 | -0.029 | 0.019 |
| opening_weight | ↑ | 75 | -0.049 | 0.019 |
| piece_values | ↓ | 61 | -0.059 | 0.018 |
| safety_weight | ↓ | 45 | -0.149 | 0.026 |

## Teste direto das tendências boas (A/B intercalado vs Stockfish 2250, contra o t0467)

| Variante | volta 1 (500) | volta 2 (1000) | média (1500) | ms/lance |
| --- | ---: | ---: | ---: | ---: |
| **t0467** | 0,568 | 0,643 | **0,618** | 7,3 / 8,8 |
| `exchange_depth=6` | 0,599 | 0,622 | 0,614 | 7,0 / 8,4 |
| `exchange_depth=7` | 0,582 | - | - | 7,3 |
| `scan_keep=0` (scan completo) | 0,606 | 0,631 | 0,623 | 7,8 / 9,2 |
| `exchange_depth=6` + `scan_keep=0` | - | 0,620 | - | 9,0 |
| `opening_weight=0.03` | 0,536 | - | - | 7,4 |
| 7 + 0,03 + scan completo | 0,597 | - | - | 7,6 |
| `check_extension=1` | 0,631 | 0,655 | **0,647** | 9,1 / 11,3 (+28%) |

Nenhuma bate o t0467 de forma clara: os ganhos da volta 1 eram ruído (±0,043). O único consistente é
`check_extension=1` (+0,029 em 1500 jogos, ~1,6 desvios), mas pensa 28% mais tempo. Fica o t0467.
