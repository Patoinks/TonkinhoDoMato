# Tonkinho do Mato — Treino

App Windows local e apenas de leitura, com progresso da campanha, resultados,
motivos de empate, ETA, ritmo recente, updates e gráficos de sobrevivência.
Atualiza a cada dois segundos; os dados novos chegam ao ritmo dos logs de treino.
Não abre checkpoints nem altera o treino. Fechar a janela não termina o treino.

`tracker_data.py` segue os JSONL incrementalmente, aguarda linhas incompletas,
agrega retomas da mesma campanha e distingue erros, interrupções e dados antigos.
A ETA usa o ritmo observado nos últimos cinco minutos. Os gráficos usam até
160 pontos e suavizam quatro registos consecutivos (normalmente 100 jogos).
Mais sobrevivência não equivale por si só a maior força de jogo.

Executável: `dist/tonkinho-tracker-build/Tonkinho do Mato - Treino/Tonkinho do Mato - Treino.exe`.
`tracker-config.json`, ao lado do executável, indica a pasta do projeto.
O botão Escolher pasta permite apontar a outro projeto durante a sessão.

Para desenvolvimento: `python desktop/tonkinho_tracker.py --project CAMINHO`.
Testes: `python -m pytest tests/test_tracker_data.py -q`.
