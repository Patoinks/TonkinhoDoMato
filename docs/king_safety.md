# Proteção do rei e pressão ofensiva

O ator V3 e o navegador usam a mesma avaliação geométrica dos dois reis.
`king_safety_weight=0.06` acrescenta uma influência suave: risco do rei adversário
menos risco do próprio rei. Zero desativa o módulo. Checkpoints antigos recebem
o valor por omissão ao serem carregados/exportados pelo código novo.

O mapa de cada rei inclui abrigo de peões nas duas filas seguintes, raios abertos,
peças que controlam a sua vizinhança, coordenação atacante, cobertura defensiva,
xeque e casas de fuga. Peças cravadas só contribuem ao longo da cravada. Para
casas de fuga, contam também os ataques de peças cravadas, como exigem as regras
dos movimentos do rei. São considerados capturas do rei e raios abertos quando
ele abandona a casa original. Roque não conta como uma casa de fuga imediata.

Ausência de abrigo só pesa se existir pressão adversária na zona do rei. A
capacidade atacante restante modula o risco (dama/torres/peças menores); estes
coeficientes medem potencial de ataque, não substituem os valores posicionais
aprendidos das peças. Um rei ativo sem ameaças num final não recebe penalização
apenas por estar exposto. Os coeficientes são heurísticos e ainda não calibrados
por resultados de jogos.

O sinal inicial ajuda a selecionar candidatos. Na busca limitada existente,
cada resposta analisada recebe nova avaliação dos dois reis; o sinal inicial é
substituído pelo da continuação, não somado outra vez. Uma defesa que neutraliza
o ataque retira o bónus. Pressão relevante força essa busca, mas mantém os mesmos
limites de candidatos e nós. O pior ramo medido participa mesmo quando o modelo
o prevê como improvável. Não se promete encontrar a melhor defesa fora do orçamento.

É uma prioridade de decisão implementada em código, não uma nova cabeça neural
treinada nem uma prova de mate/compensação por sacrifício. A rede e o objetivo de
sobrevivência permanecem iguais. Ataques podem ser escolhidos mesmo envolvendo
perda material se a avaliação combinada os favorecer. Futuras experiências V3
usam esta política e guardam o mapa da jogada escolhida no traço, mas o módulo não
atribui recompensa extra por dar xeque e não lança treino automaticamente.

Limites: cobertura geométrica não prova que um defensor consiga responder a
todas as ameaças simultâneas; sobrecarga, desvios e combinações longas dependem
da busca e da rede. A avaliação não é uma probabilidade de mate. Medir força
exige jogos de avaliação separados; os testes validam mecânica e equivalência.

Validação local: 45 testes Python dos módulos cognitivo, peças e rei; 21 posições
de referência com decisões e mapas equivalentes no navegador. O benchmark
`perf_results/king_safety_benchmark.py` usa 16 posições do checkpoint concluído,
uma passagem de aquecimento e duas medidas, CPU com uma thread, mantendo a
proteção das peças ativa. Nesta máquina mediu 22,75 ms sem o módulo e 30,17 ms
com o módulo por decisão em média (medianas 19,58 e 29,50 ms). Não mede jogos por
segundo nem força. Os números dependem da posição e do orçamento de busca.
