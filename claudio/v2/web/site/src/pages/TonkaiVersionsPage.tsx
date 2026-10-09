import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BOT_NAME } from '../tonkai/brand';
import './TonkaiVersionsPage.css';

const versions = [
  {
    id: 'v0', name: 'O ponto de partida', phase: 'Primeira rede',
    change: 'Aprender com o resultado das partidas, sem copiar um motor de xadrez.',
    details: [
      'Avalia vitória, empate e derrota para escolher entre jogadas legais.',
      'Tenta prever a próxima jogada do adversário e estima o valor de cada peça segundo a posição.',
      'Treina contra o TinyOpponent. O adversário inicial era forte demais e quase todas as partidas terminavam em derrota, deixando pouco sinal útil para aprender.',
    ],
  },
  {
    id: 'v1', name: 'Aprender com adversários ajustados', phase: 'Fase intermédia',
    change: 'O treino passa a receber resultados variados em vez de derrotas quase constantes.',
    details: [
      'Um currículo ajusta a dificuldade do TinyOpponent numa escala de 12 níveis conforme os resultados do Tonkinho.',
      'São acrescentados históricos de treino, ferramentas para jogar no terminal e calibração opcional do adversário.',
      '“v1” identifica aqui a evolução entre o projeto inicial e a V2; o repositório não contém um lançamento formal com esse nome.',
    ],
  },
  {
    id: 'v2', name: 'Memória e sobrevivência', phase: 'Novo treino',
    change: 'Deixa de olhar apenas para a posição atual e aprende com sequências de jogadas.',
    details: [
      'Uma memória temporal acompanha a partida; a rede prevê posições futuras e revê episódios surpreendentes.',
      'Treina com adversários diferentes, incluindo versões anteriores de si próprio e um Stockfish limitado.',
      'O objetivo principal é evitar a derrota; vitória e empate têm o mesmo alvo de sobrevivência nesta fase.',
    ],
  },
  {
    id: 'v3', name: 'Analisar antes de decidir', phase: 'Cognição e v3.1',
    change: 'Compara respostas possíveis e usa experiências passadas para escolher melhor.',
    details: [
      'Acrescenta memória de partidas, experiências com jogadas alternativas, análise adaptativa e atenção à pior resposta encontrada.',
      'A evolução v3.1 introduz um objetivo que distingue vitória de empate, mede capturas e melhora a lógica de desistência e repetição.',
      'O cálculo exato de mates curtos foi acrescentado depois: procura ataques e defesas forçadas antes de confiar na estimativa da rede.',
    ],
  },
  {
    id: 'v4', name: 'Rei e peças em perigo', phase: 'Versão anterior do site',
    change: 'Aprende a reconhecer ameaças de mate e passa a usar valores de peças medidos em partidas.',
    details: [
      'Uma rede de segurança do rei aprende a partir de posições com mate forçado, sem uma tabela fixa de abrigos ou padrões.',
      'O jogo combina essa estimativa com cálculo de mates e com valores contextuais das peças, treinados a partir dos resultados de capturas.',
      'Foi a versão do Tonkinho que esteve no tabuleiro antes do G16 v2.',
    ],
  },
  {
    id: 'v5', name: 'Criar jogo com proteção', phase: 'Experimental',
    change: 'Procura pressão favorável e mais mobilidade sem deixar as próprias peças desprotegidas.',
    details: [
      'Compara a soma dos valores aprendidos dos atacantes com a dos defensores que podem recapturar legalmente.',
      'Procura ameaças de uma peça menos valiosa sobre outra mais valiosa quando a casa de chegada fica protegida.',
      'Divide o novo sinal de atividade em 50% ataque/proteção e 50% mobilidade segura. O checkpoint treinou mais 500 partidas de self-play.',
      'Num confronto de 100 partidas contra a v4, obteve 43 vitórias, 26 empates e 31 derrotas (56% dos pontos). Ainda não está publicado no tabuleiro.',
    ],
    experimental: true,
  },
  {
    id: 'g16-v2', name: 'G16 v2', phase: 'Disponível para jogar',
    change: 'O Claudio completo passa a jogar no navegador, com a rede do checkpoint 600 e os módulos do v2.',
    details: [
      'Usa a rede compacta, memória da partida, livro de experiência próprio e política aprendida.',
      'Calcula as trocas com os valores das peças que mediu por fase e procura mates forçados para converter finais.',
      'O motor C++ corre no navegador através de WebAssembly. Os empates por regras terminam a partida e valem 0,5 pontos.',
      'Na comparação com o G16 original, ganhou 15 partidas, perdeu 12 e houve 3 partidas inacabadas ao limite de lances. Não houve empates efetivos nesse confronto.',
    ],
    live: true,
  },
];

export default function TonkaiVersionsPage() {
  useEffect(() => {
    const previous = document.title;
    document.title = `Versões do ${BOT_NAME} · ANW`;
    return () => { document.title = previous; };
  }, []);

  return <div className="tkv-page">
    <header className="tkv-hero">
      <Link className="tkv-back" to="/tonkai">← Voltar ao tabuleiro</Link>
      <p className="tkv-eyebrow">{BOT_NAME.toUpperCase()} · EVOLUÇÃO</p>
      <h1>Da primeira jogada<br /><span>à criação de jogo.</span></h1>
      <p className="tkv-lead">Da rede inicial ao G16 v2. Cada versão acrescenta uma forma de aprender, avaliar ou escolher jogadas.</p>
      <div className="tkv-status"><span><i className="tkv-dot" /> No site: G16 v2</span></div>
    </header>

    <nav className="tkv-jump" aria-label="Ir para uma versão">
      {versions.map(version => <a key={version.id} href={`#${version.id}`}>{version.id}</a>)}
    </nav>

    <main className="tkv-timeline" aria-label="Evolução das versões">
      {versions.map((version, index) => <section className="tkv-entry" id={version.id} key={version.id} aria-labelledby={`${version.id}-title`}>
        <div className="tkv-marker"><span>{version.id}</span></div>
        <div className="tkv-card">
          <div className="tkv-card-top"><span>{String(index + 1).padStart(2, '0')} / {String(versions.length).padStart(2, '0')} · {version.phase}</span>{version.live && <strong className="tkv-badge">A jogar no site</strong>}{version.experimental && <strong className="tkv-badge tkv-badge-experimental">Em teste</strong>}</div>
          <h2 id={`${version.id}-title`}>{version.name}</h2>
          <p className="tkv-change">{version.change}</p>
          <ul>{version.details.map(detail => <li key={detail}>{detail}</li>)}</ul>
        </div>
      </section>)}
    </main>

    <footer className="tkv-end"><p>Queres experimentar a versão disponível?</p><Link to="/tonkai">Jogar contra o G16 v2 <span aria-hidden="true">→</span></Link></footer>
    <p className="tkv-piece-credits">Peças <a href="https://github.com/Kadagaden/chess-pieces">Kaneo por Kadagaden (Ka-hu)</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></p>
  </div>;
}
