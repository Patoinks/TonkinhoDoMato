# Oponente UCI leve: Stockfish 11, 1600 Elo nominal

Este adaptador fornece apenas lances de um adversário externo. Não fornece
avaliações, centipawns ou alvos de treino. O treino legado e o TinyOpponent não
são alterados.

## Motor e origem

- **Stockfish 11 64**, Windows x64 genérico, avaliação clássica, sem ficheiro
  NNUE externo. Executável de **1.172.480 bytes**; arquivo ZIP de **2.004.257 bytes**.
- [Anúncio oficial da versão 11](https://stockfishchess.org/blog/2020/stockfish-11/).
- A [página oficial de downloads](https://stockfishchess.org/download/) liga o
  [arquivo de versões antigas](https://drive.google.com/drive/folders/1nzrHOyZMFm4LATjF5ToRttCU0rHXGkXI).
  Na [pasta Stockfish 11](https://drive.google.com/drive/folders/10b915Xd9DcTvkAVPib9VETcJtRUkK14U),
  foi usado `stockfish-11-win.zip`, ficheiro `1it9o-AF6jb55z02Xvu-Uu387Q_CPa0Ed`.
- O ZIP inclui `Copying.txt`, `Readme.md` e `src/`. O código identifica a licença
  **GPL-3.0-or-later**. Mantemos estes ficheiros ao lado do executável; o
  [código oficial sf_11](https://github.com/official-stockfish/Stockfish/tree/sf_11)
  também está disponível. Não incorporamos o motor no módulo Python.
- O UCI real anunciou `UCI_Elo` entre **1350 e 2850**, `Threads` entre 1 e 512,
  `Hash` entre 1 e 131072 e `UCI_LimitStrength`. O adaptador verifica as opções
  anunciadas e rejeita valores incompatíveis; nunca cai silenciosamente para
  força máxima.

SHA256 do ZIP:

```text
d5a6149d4174e4f8a96011f67cb08f929e135d367e20d83456a4ff29e1e5133a
```

SHA256 do executável x64 genérico:

```text
95ccc04678193f8f8e7e595de49a3e5270f5e853aa89a147061d65a3dbe59d81
```

Estes hashes foram medidos no arquivo obtido pelo percurso oficial e ficam
fixados no instalador para reproduzir o mesmo artefacto. Não são apresentados
como uma assinatura digital ou um checksum publicado pelo autor.

## Instalar e usar

Na raiz do repositório, em PowerShell:

```powershell
& .\scripts\setup_opponent.ps1
python -m pytest tests/test_uci_opponent.py -q
```

O instalador verifica o ZIP antes de extrair e o executável depois de extrair.
Conserva a licença e o código-fonte. `engines/` está ignorado pelo Git.
O executável fica em:

```text
engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64.exe
```

```python
import chess
from tonkai.uci_opponent import UCIOpponent

path = "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64.exe"
with UCIOpponent(path, elo=1600, move_time=0.02, threads=1, hash_mb=16) as opponent:
    print(opponent.description)
    print(opponent.info)  # Configuração e proveniência, sem avaliação da posição.
    opponent.reset()
    board = chess.Board()
    move = opponent.choose(board)
    board.push(move)
```

Uma instância mantém **um processo persistente** e deve pertencer a um único
ator. Passa-se o `chess.Board` com o seu `move_stack`; reconstruir apenas a FEN
perderia o histórico de repetição. `choose` não modifica o tabuleiro e rejeita
posições inválidas, partidas terminadas e respostas ilegais. `rng` é aceite por
compatibilidade, mas não controla a aleatoriedade interna do Stockfish: a
mesma semente do treino não garante as mesmas partidas.

Chamar `reset()` antes de cada partida. O token de jogo muda e python-chess
envia `ucinewgame` antes da próxima pesquisa. `close()` é idempotente; o context
manager também encerra o processo se o código do chamador falhar. Um erro ou
timeout durante a escolha fecha o motor e é propagado, sem jogada aleatória
de substituição. O parâmetro opcional `timeout=5.0` limita a espera adicional
do protocolo: com `move_time=0.02`, a pesquisa pode esperar até 5,02 segundos
por um motor que não responde. `move_time` tem resolução UCI de milissegundos.

Usamos `Limit(time=move_time)`, `INFO_NONE` e `ponder=False`. Não chamamos
`analyse`, nem recolhemos pontuações como rótulos. Os valores padrão são
`UCI_LimitStrength=true`, `UCI_Elo=1600`, `Threads=1`, `Hash=16`.

## O que significa 1600

O [README da versão 11](https://github.com/official-stockfish/Stockfish/blob/sf_11/Readme.md#uci-parameters)
documenta a calibração de `UCI_Elo` a **60 s + 0,6 s**, ancorada à escala
CCRL 40/4. **20 ms por lance não reproduz essa calibração**, e 1600 nesta opção
não prova 1600 FIDE, Chess.com ou Lichess. Hardware, limite de tempo e versão
do motor afetam a força observada. Para comparar treinos, fixar estas três
coisas e guardar `opponent.info` no relatório/checkpoint do chamador.

O controlo de força seleciona lances deliberadamente mais fracos. Isso cria um
adversário com erros; não significa que o treino deva copiar os seus valores
materiais ou usar as suas avaliações como professor.

## Verificação local, 22/09/2026

Windows 11, Python 3.13.14, python-chess/chess 1.11.2. Os **30 testes** passaram,
incluindo execução real, preservação do histórico, limites das opções,
respostas ilegais, timeout e encerramento.

Um smoke separado fez **20 lances consecutivos**, todos legais, desde a posição
inicial, com o mesmo processo e configuração padrão. Latência total de
`choose`, incluindo o adaptador e a comunicação UCI:

| Medida | Tempo |
|---|---:|
| Média | 20,754 ms |
| Mediana | 20,776 ms |
| Mínimo / máximo | 19,919 / 22,366 ms |
| Arranque e configuração | 59,428 ms |
| Encerramento | 2,924 ms |

Os detalhes desta execução estão em `engines/opponent_1600_smoke.json` local
(ignorado). A medição verifica funcionamento/latência, **não estima Elo** nem
o desempenho do treino completo. Para repetir o smoke legal, o teste
`test_real_engine_twenty_legal_plies` usa o executável instalado; sem motor,
é saltado. `TONKAI_UCI_TEST_ENGINE` permite fornecer outro caminho explicitamente.
