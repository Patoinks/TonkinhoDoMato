# Sensação de troca e proteção por peça

O ator V3 e o navegador calculam um estado para todas as peças, antes de escolher
quais as jogadas que vão analisar mais profundamente. Este estado afeta a escolha,
não é apenas um registo. Não inicia nem prolonga treinos existentes.

1. **Troca:** calcula o saldo de uma sequência local de capturas e recapturas.
   Uma perda superior a 0,5 é o primeiro alerta. Uma troca favorável acima dessa
   margem não exige proteção adicional. Cada lado pode parar de recapturar.
2. **Proteção:** dentro de ±0,5 verifica quem consegue recapturar legalmente após
   a captura adversária. Uma peça cravada que não consiga mexer não conta.
3. **Quantidade:** compara atacantes distintos com defensores legais. Dois
   atacantes e um defensor geram défice de proteção; dois defensores satisfazem
   esse critério quantitativo. Isso não prova que a posição seja segura: o saldo
   da troca continua a ter prioridade.

Os valores vêm da `piece_head` posicional já aprendida. Não se introduziu uma
tabela fixa de peão=1/cavalo=3/dama=9. Para a margem 0,5 ter escala utilizável,
o valor mediano absoluto dos peões da posição define uma unidade local; há um
piso de 0,001 em unidades de sobrevivência para posições sem peões ou previsões
quase nulas, e valores normalizados limitados a ±12. **Esta unidade não é um
centipeão calibrado**. Os valores ainda podem ser imprecisos e até negativos;
melhoram apenas se a cabeça aprendida melhorar com treino.

A verificação atualiza ocupação, cravadas, xeques e raios de ataque em cada
recaptura, e trata en passant e promoções. O rei recebe tratamento de xeque e
nunca é avaliado como material capturável. A sequência tem um limite de seis
capturas e segue o recapturador legal de menor valor aprendido. É uma aproximação
local, não uma busca completa de combinações, desvios ou defesas sobrecarregadas
em casas diferentes. Os defensores são condicionais a cada captura; não se somam
defensores incompatíveis de ramificações diferentes.

A influência é suave (`piece_safety_weight=0.08`), preservando a possibilidade
de um sacrifício que a avaliação de sobrevivência considere compensado. Continua
a existir exploração durante o treino. O objetivo de sobrevivência não mudou.
Os traços guardam riscos próprios/adversários sem duplicar o mapa inteiro em cada
experiência. `agent.sense_pieces(board, colour)` expõe o detalhe por casa.
