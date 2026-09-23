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

A aba **Score por adversário** mostra jogos, vitórias, empates, derrotas,
partidas no limite, pontos e score por adversário na campanha atual. Diferentes
níveis de Elo nominal/configurações de um motor ficam separados. Selecionar uma
linha mostra a percentagem de vitórias, jogos concluídos, desistências registadas
e tempo por jogada quando disponível. Self-play tem uma linha própria e mantém
a distinção dos totais externos usada pelo treino.

Pontos = vitórias + 0,5 × empates. Score = pontos / (vitórias + empates + derrotas).
Desistências contam como derrotas; jogos truncados não entram no denominador.
Sem jogos concluídos, o score aparece como «—». O score não é uma estimativa de Elo.

Lê os contadores `metrics.opponents` por sessão (`self`, `tiny`, `fairy`, `external`)
e a configuração do registo `run`. Nas retomas, conserva o prefixo do checkpoint
e descarta jogos posteriores que não foram gravados. Logs antigos de um único
motor permitem recuperar a separação pelas diferenças dos contadores; resultados
sem atribuição verificável aparecem em **Sem adversário registado**. Não inventa
uma divisão para misturas antigas sem contadores por adversário.

A aba **Ritmo e duração** conserva os gráficos. A app acompanha o objetivo
registado no treino, em vez de mostrar sempre «sobrevivência».

Executável: `dist/tonkinho-tracker-build/Tonkinho do Mato - Treino/Tonkinho do Mato - Treino.exe`.
`tracker-config.json`, ao lado do executável, indica a pasta do projeto.
O botão Escolher pasta permite apontar a outro projeto durante a sessão.

Para desenvolvimento: `python desktop/tonkinho_tracker.py --project CAMINHO`.
Testes: `python -m pytest tests/test_tracker_data.py -q`.


A aba **Todas as estatísticas** apresenta todos os campos do último registo de
métricas, incluindo perdas por cabeça, capturas/KD, calibração, experiências,
pesquisa, desistências e métricas de execução. Mostra também configuração e
resumos da campanha. A pesquisa filtra por nome/caminho ou valor, e campos novos
aparecem automaticamente. Valores zero e null não são escondidos; dados não
registados não são inventados. Os campos em `treino` são o último registo da
sessão; os de `campanha` agregam retomas quando aplicável.

As métricas `execution` identificam CPU/GPU, opções ativas, tempo por update,
updates e posições aprendidas na sessão, tempo agregado dos atores e memória
CUDA alocada (quando disponível). Os tempos dos atores somam trabalho paralelo,
pelo que não equivalem ao tempo de parede da campanha.
