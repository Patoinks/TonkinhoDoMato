# AI Tier Maker no ANW Games

Origem: `C:\Users\Martim\Tonkai\Tools\AiTierMaker`. A pasta original fica intacta.
Página estática em `/ai-tier-maker/`, com `<base>` para suportar também o URL sem barra final.

`public/` contém a cópia publicável, a navegação para as restantes páginas e a data da captura fornecida. O seletor `.ghost` foi limitado a `.card.ghost`: anteriormente também apanhava os botões, tornando-os fixos e sem eventos de ponteiro.

O script `site-navigation.js` acrescenta uma ligação de documento aos menus desktop/mobile das aplicações React existentes. É carregado nos seus HTML de entrada, sem reconstruir nem substituir bundles locais que contêm trabalho em curso. `source.patch` regista as duas alterações aos HTML fonte do checkout ANW; os novos recursos também ficam em `public/ai-tier-maker/` desse checkout.

Publicação seletiva com o login Firebase existente:

```powershell
node site-patches/ai-tier-maker/publish.mjs --prepare
node site-patches/ai-tier-maker/smoke.mjs
node site-patches/ai-tier-maker/publish.mjs --publish
node site-patches/ai-tier-maker/publish.mjs --verify
node site-patches/ai-tier-maker/smoke.mjs --live
```

O publicador mantém todos os hashes live, exceto os HTML onde injeta a navegação, e adiciona os recursos do Tier Maker. Preserva as regras existentes, acrescenta apenas o rewrite e o cabeçalho desta rota e recusa ativar uma versão se outro agente publicar entretanto. `evidence/` guarda manifestos e resultados; renders e ficheiros temporários não entram no Git.

O smoke testa os menus de todas as aplicações, os 20 cartões/logótipos, rato/teclado, persistência, ordenação automática, criação/remoção de tiers, toggle de ranking, descarga PNG real e layout até 320 px. A prévia interceta apenas os ficheiros preparados no navegador headless; os bundles restantes são os já publicados.
