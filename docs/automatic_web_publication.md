# Publicação automática do modelo jogável

Ativada nesta máquina por `web_publish.local.json`. No fim normal de `train.py`,
`train_survival.py` ou `train_cognitive.py`, depois de fechar os atores e guardar
o checkpoint final, o hook chama `scripts/publish_web_model.py`.

Só checkpoints dentro das pastas autorizadas (por defeito `checkpoints/`)
entram no processo. Treinos interrompidos, falhados, simulações `--no-train` e
testes em pastas temporárias não publicam. Para desligar, definir `enabled`
como `false` no ficheiro local. Não altera o objetivo nem o limite de jogos.

O processo guarda uma cópia imutável, exporta os pesos e o estado de inferência,
compara avaliações/jogadas com o Python, corre testes de regras, compila o site,
testa o navegador e o bundle de produção e publica no Firebase Hosting. Confirma
depois o manifesto e SHA256 dos pesos descarregados do site. Os logs e o último
resultado ficam em `history/web-publications/`; o checkpoint do treino permanece
guardado mesmo que a publicação falhe. Erros da publicação são anunciados no
terminal; falhas do deploy são tentadas três vezes.

O V3 no navegador usa GRU com histórico real, ensemble de sobrevivência, memória
associativa com encoder congelado, calibração de erro, procura robusta limitada
e planos aprendidos. Exploração aleatória e experiências contrafactuais pertencem
ao treino; o navegador joga deterministicamente. A saída de sobrevivência não é
apresentada como uma probabilidade W/D/L. V2 e o modelo legado também são aceites.

Os pesos recebem um nome contendo o hash. A página verifica uma pequena versão
a cada 30 segundos e ao recuperar foco. Uma partida já iniciada conserva a sua
versão; a próxima usa a versão nova. O painel do treino e o modelo jogável têm
publicações distintas, coordenadas por um lock para impedir que uma atualização
de estatísticas publique um candidato ainda em validação.

Repetir uma publicação falhada, sem voltar a treinar:

```powershell
python scripts/publish_web_model.py --checkpoint checkpoints/cognitive-1000-500/latest.pt
```

Requer o computador ligado, Python/Node/Chrome instalados, acesso à Internet e a
sessão local do Firebase válida. Após encerramento abrupto durante uma publicação,
verificar o PID registado em `history/web-publications/publish.lock` antes de
remover um lock órfão. Não há um serviço remoto a treinar/publicar com o PC desligado.
