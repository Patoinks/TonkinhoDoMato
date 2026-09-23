# Quadro de coordenação

Mais do que um agente trabalha neste repositório ao mesmo tempo (Claude e ChatGPT/Codex).
Este quadro evita que um apague o trabalho do outro.

## Regras

1. **Antes de editar um ficheiro, reserva-o aqui.** Acrescenta uma linha em "Reservas ativas".
2. **Não edites um ficheiro reservado por outro agente.** Se precisares dele, escreve em "Pedidos" e espera.
3. **Ficheiros novos também se reservam.** Reserva-os antes de os criares.
4. **Quando acabares:**
   - corre os testes (`python -m pytest -q`);
   - faz commit só dos teus ficheiros (`git add <ficheiros>`, nunca `git add -A` nem `git add .`);
   - apaga as tuas reservas e regista em "Concluído".
5. **Nunca faças `git checkout`, `git reset`, `git stash` nem `git restore` a ficheiros que não reservaste.**
   Apagariam o trabalho por gravar de outro agente.
6. **Lê este ficheiro de novo antes de cada nova edição.** Não confies no que leste há 10 minutos.
7. **Não relances nem pares o treino sem o Martim pedir.**

Formato de uma reserva: `- <agente> | <ficheiros ou pasta> | <o que está a fazer> | <data hora>`

## Reservas ativas

- Claude | tonkai/opponents.py, tonkai/cognitive_config.py, tonkai/cognitive_pool.py, train_cognitive.py, tests/test_opponents.py | mistura de adversários (self-play/Tiny/Fairy/Stockfish) + separação de resultados por adversário no treino | 2026-09-23
- Claude | tonkai/survival_trainer.py, tonkai/cognitive_trainer.py, tonkai/cognitive_agent.py, tonkai/cognitive_memory.py, tonkai/temporal_model.py | 7 melhorias de velocidade do treino/pensamento | 2026-09-23
- ChatGPT | desktop/tonkinho_tracker.py, desktop/tracker_data.py, tests/test_tracker_data.py | app de acompanhamento: separar valores por adversário | 2026-09-23 (reserva feita pelo Claude em nome dele, pelo que estava no disco; o ChatGPT deve confirmar)

## Pedidos

(nenhum)

## Concluído

(nada ainda)
