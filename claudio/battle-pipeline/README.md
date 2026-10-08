# Pipeline de batalhas

Este runner faz batalhas repetíveis entre os checkpoints em `registry.json`. Guarda cada jogo e atualiza a tabela durante a execução; `runs/` fica local e é retomável.

## Primeira utilização neste PC

O runner precisa de Python 3.13, PyTorch, NumPy e `python-chess`. Compila uma vez a pesquisa nativa otimizada no Windows/MSVC:

```powershell
Push-Location claudio/fast
$env:CLAUDIO_NATIVE_ARCH = 'AVX2'
python setup.py build_ext --inplace --force
Remove-Item Env:\CLAUDIO_NATIVE_ARCH
Pop-Location
```

Se o Python não estiver na instalação normal do Python 3.13, define `$env:TONKAI_PYTHON` para o caminho do `python.exe`. As partidas usam CPU; a GPU fica para treino.

## Comandos

```powershell
# A batalha mais comum
.\claudio\battle-pipeline\battle.ps1 -A G16 -B G15Treinado -Games 30

# Lista os nomes e atalhos
.\claudio\battle-pipeline\battle.ps1 -List

# Uma fase entre três AIs: 3 pares × 20 partidas = 60
.\claudio\battle-pipeline\battle.ps1 -Players G16,G15Treinado,Forte -Games 20

# Todos os 15 participantes registados: 105 pares × 20 = 2100 partidas
.\claudio\battle-pipeline\battle.ps1 -All -Games 20

# Estado, retoma, paragem graciosa e acompanhamento
.\claudio\battle-pipeline\battle.ps1 -Status
.\claudio\battle-pipeline\battle.ps1 -Resume
.\claudio\battle-pipeline\battle.ps1 -Stop
.\claudio\battle-pipeline\battle.ps1 -A G16 -B G15Treinado -Games 30 -Watch
```

Usa dois workers com prioridade baixa por omissão; aceita de um a seis. Cada par recebe aberturas emparelhadas, as cores alternam e há quatro meios-lances aleatórios. O limite normal é 300 meios-lances. Jogos que chegam ao limite aparecem como **inacabados**, separados dos empates. A tabela define a taxa de vitórias como vitórias divididas por todos os jogos.

Os especialistas `Cores`, `CoresNotas` e `CoresRev5` usam o checkpoint de brancas quando jogam de brancas e o de pretas quando jogam de pretas. Stockfish 11 é externo ao repositório: para o incluir, coloca o executável original em `engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe`. A pipeline valida o SHA registado. “Stockfish Max” e “Stockfish 1800” usam esse mesmo executável e o mesmo limite de 2 ms por lance; Max desativa o limite de Elo.

Cada execução cria uma pasta em `runs/`, atualiza `LATEST.txt` e guarda `manifest.json`, `status.json`, `games.jsonl`, PGN, `standings.md` e `results.json`. O manifesto regista hashes, opções e tempos. Não copia checkpoints nem compila o código durante o arranque normal.
