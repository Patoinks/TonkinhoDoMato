import { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess, type Color, type Square } from 'chess.js';
import { Game } from '../tonkai/game';
import { loadNetwork, type Network } from '../tonkai/network';
import { saveGame } from '../tonkai/data';
import { createWinRecord, fetchFeaturedGame, saveLocalLongestWin, type FeaturedGame } from '../tonkai/winHistory';
import './TonkaiPage.css';
import { AuthContext } from '../context/auth';
import { BOT_AVATAR } from '../tonkai/brand';
import { pieceValueLabel, positionEvaluation } from '../tonkai/positionEvaluation';
import { useStockfishEvaluation } from '../tonkai/useStockfishEvaluation';
import TonkaiRoastChat from './TonkaiRoastChat';
import TonkaiEvalBars from './TonkaiEvalBars';
import TonkaiMoveIcon from './TonkaiMoveIcon';
import { MOVE_GRADES, reviewMove, reviewDescription } from '../tonkai/moveReview';
import type { RoastContext } from '../tonkai/roastContext';
import { TONKAI_PIECES, TonkaiPiece } from './TonkaiPieces';
import { BLITZ_TIME_CONTROL, type ChessClock } from '../tonkai/clock';
import TonkaiClock from './TonkaiClock';

function PlayerIdentity({ name, color, bot = false, photo, active = false, clock, flagged = false }: {
  name: string; color: Color; bot?: boolean; photo?: string | null; active?: boolean; clock?: ChessClock | null; flagged?: boolean;
}) {
  const [failedPhoto, setFailedPhoto] = useState<string | null>(null);
  const image = bot ? BOT_AVATAR : photo;
  return <div className={`tk-player${active ? ' tk-player-active' : ''}`} aria-label={`Jogador: ${name}`}>
    <div className={`tk-avatar${bot ? ' tk-avatar-bot' : ''}`}>
      {image && failedPhoto !== image ? <img src={image} alt={`Avatar de ${name}`} width="52" height="52" onError={() => setFailedPhoto(image)} />
        : <span aria-hidden="true">{name.trim().charAt(0).toUpperCase()}</span>}
    </div>
    <div className="tk-player-info"><strong>{name}</strong><span><i className={`tk-piece-color tk-piece-${color}`} aria-hidden="true" />{color === 'w' ? 'Brancas' : 'Pretas'}{bot && <small>IA</small>}</span></div>
    {clock !== undefined ? <TonkaiClock clock={clock} color={color} active={active} flagged={flagged} />
      : active && <span className="tk-player-turn" aria-label="A jogar" />}
  </div>;
}

const services = { saveGame };
const BOT_NAME = 'G16 v2';
const MODEL_MANIFEST = '/tonkai/g16-v2.json';
export default function TonkaiPage({ initialFen, persistence = services }: {
  initialFen?: string; persistence?: typeof services;
}) {
  const auth = useContext(AuthContext);
  const playerName = auth?.profile?.displayName || auth?.profile?.username || auth?.user?.displayName || 'Tu';
  const [network, setNetwork] = useState<Network | null>(null);
  const [loadError, setLoadError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [game, setGame] = useState<Game | null>(null);
  const [revision, setRevision] = useState(0);
  const [colour, setColour] = useState<Color>('w');
  const [selected, setSelected] = useState<string | null>(null);
  const [promotion, setPromotion] = useState<{ from: string; to: string } | null>(null);
  const [resultDismissed, setResultDismissed] = useState(false);
  const [playError, setPlayError] = useState('');
  const [view, setView] = useState<'play' | 'history'>('play');
  const [featured, setFeatured] = useState<FeaturedGame | null>(null);
  const [replayPly, setReplayPly] = useState(0);
  const [showEvaluations, setShowEvaluations] = useState(false);
  const gameId = useRef('');
  const submitted = useRef(false);
  const replayWin = featured;

  useEffect(() => {
    const previous = document.title;
    document.title = `${BOT_NAME} · ANW`;
    return () => { document.title = previous; };
  }, []);

  useEffect(() => {
    let active = true;
    setLoadError('');
    loadNetwork(MODEL_MANIFEST).then(n => { if (active) setNetwork(n); })
      .catch(e => { if (active) setLoadError(String(e.message)); });
    return () => { active = false; };
  }, [attempt]);

  useEffect(() => {
    if (!network) return;
    let active = true, loading = false;
    const refreshModel = async () => {
      if (loading) return;
      loading = true;
      try {
        const response = await fetch('/tonkai/g16-v2-version.json', { cache: 'no-cache' });
        if (!response.ok) return;
        const version = await response.json();
        if ((version.revision ?? version.sha256) !== (network.manifest.revision ?? network.manifest.sha256)) {
          const next = await loadNetwork(MODEL_MANIFEST);
          // Existing Game retains its network; the next game uses the new model.
          if (active) setNetwork(next);
        }
      } catch { /* Keep the validated local model while offline. */ }
      finally { loading = false; }
    };
    const timer = window.setInterval(() => { void refreshModel(); }, 30_000);
    window.addEventListener('focus', refreshModel);
    return () => { active = false; window.clearInterval(timer); window.removeEventListener('focus', refreshModel); };
  }, [network]);

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const game = await fetchFeaturedGame();
        if (active) setFeatured(game);
      } catch {
        // Keep the last available replay if refreshing fails.
      }
    }
    void refresh();
    window.addEventListener('focus', refresh);
    return () => { active = false; window.removeEventListener('focus', refresh); };
  }, [view]);

  useEffect(() => { setReplayPly(0); }, [replayWin?.pgn]);

  useEffect(() => {
    if (!game || game.over || game.humanTurn || playError) return;
    const timer = window.setTimeout(() => {
      try { game.playTonkAI(); setRevision(n => n + 1); }
      catch { game.clock?.stop(); setPlayError('Não foi possível jogar esta posição. Inicia uma nova partida.'); }
    }, 160);
    return () => window.clearTimeout(timer);
  }, [game, revision, playError]);

  useEffect(() => {
    if (!game?.clock || game.over || playError) return;
    const check = () => {
      if (!game.checkClock()) return;
      setSelected(null); setPromotion(null); setRevision(n => n + 1);
    };
    const timer = window.setInterval(check, 100);
    window.addEventListener('focus', check);
    document.addEventListener('visibilitychange', check);
    return () => {
      window.clearInterval(timer); window.removeEventListener('focus', check);
      document.removeEventListener('visibilitychange', check);
    };
  }, [game, revision, playError]);

  useEffect(() => {
    if (!game?.over || submitted.current) return;
    submitted.current = true;
    // Keep the playable win locally even when community statistics are unavailable.
    const win = createWinRecord(game);
    if (win) saveLocalLongestWin(win);
    void persistence.saveGame(gameId.current, game.finish()).catch(error => {
      console.warn('Could not save TonkAI community statistics.', error);
    });
  }, [game, revision, persistence]);

  function start() {
    if (!network) return;
    gameId.current = crypto.randomUUID(); submitted.current = false;
    setGame(new Game(network, colour, initialFen, BLITZ_TIME_CONTROL)); setSelected(null); setPromotion(null); setResultDismissed(false);
    setPlayError('');
  }
  function commitMove(uci: string) {
    if (!game) return false;
    if (!game.playHuman(uci)) {
      if (game.over) { setSelected(null); setPromotion(null); setRevision(n => n + 1); }
      return false;
    }
    setSelected(null); setPromotion(null);
    setRevision(n => n + 1); return true;
  }
  function tryMove(from: string, to: string) {
    if (view !== 'play' || !game?.humanTurn || promotion || playError) return false;
    const moves = game.board.moves({ verbose: true }).filter(m => m.from === from && m.to === to);
    if (!moves.length) return false;
    if (moves.some(m => m.promotion)) { setPromotion({ from, to }); return false; }
    return commitMove(from + to);
  }
  const activeSelection = view === 'play' && game?.humanTurn && !promotion && !playError ? selected : null;
  const available = activeSelection && game
    ? game.board.moves({ square: activeSelection as Square, verbose: true }) : [];
  const destinations = new Map(available.map(move => [move.to, !!move.captured]));
  const status = !game ? 'Pronto para jogar' : game.over
    ? game.result === 'draw' ? 'Empate' : game.result === 'win' ? `${BOT_NAME} venceu` : 'Ganhaste'
    : game.humanTurn ? `A tua vez${game.board.isCheck() ? ' · Xeque' : ''}` : `${BOT_NAME} a jogar…`;
  const replay = view === 'history';
  const currentPly = Math.min(replayPly, replayWin?.plies ?? 0);
  const position = replay ? replayWin?.fens[currentPly] : game?.board.fen();
  const bottomColor: Color = replay ? replayWin?.tonkaiColor ?? 'w' : game?.human ?? colour;
  const topColor: Color = bottomColor === 'w' ? 'b' : 'w';
  const turn = position?.split(' ')[1] as Color | undefined;
  const playing = !replay && !!game && !game.over && !playError;
  const resultDetail = !game?.over ? '' : game.timeoutColor
    ? game.result === 'draw' ? 'Tempo esgotado. O outro jogador só tem o rei.'
      : game.timeoutColor === game.human ? 'O teu tempo esgotou.' : `O tempo do ${BOT_NAME} esgotou.`
    : game.resigned ? 'Desististe da partida.'
    : game.board.isCheckmate() ? 'Xeque-mate.'
    : game.board.isStalemate() ? 'Afogamento.'
    : game.board.isThreefoldRepetition() ? 'Repetição de posição.'
    : game.board.isInsufficientMaterial() ? 'Material insuficiente.'
    : game.board.isDrawByFiftyMoves() || Number(game.board.fen().split(' ')[4]) >= 99 ? 'Regra dos 50 lances.' : 'A partida terminou.';
  const evaluation = useMemo(() => {
    try {
      if (view === 'history') {
        if (!network || !replayWin) return null;
        const replayGame = new Game(network, replayWin.tonkaiColor === 'w' ? 'b' : 'w', replayWin.fens[0]);
        for (const move of replayWin.moves.slice(0, currentPly)) replayGame.board.move(move);
        return positionEvaluation(replayGame);
      }
      return game ? positionEvaluation(game) : null;
    }
    catch { return null; }
  }, [game, revision, view, network, replayWin, currentPly]);
  const reviewedMoves = useMemo(() => {
    if (!replay) return game?.board.history({ verbose: true }) ?? [];
    if (!replayWin) return [];
    const board = new Chess(replayWin.fens[0]);
    for (const move of replayWin.moves.slice(0, currentPly)) board.move(move);
    return board.history({ verbose: true });
  }, [replay, replayWin, currentPly, game, revision]);
  const reviewFens = useMemo(() => [...new Set(reviewedMoves.flatMap(move => [move.before, move.after]))].reverse(), [reviewedMoves]);
  const { evaluation: stockfish, reviews, error: stockfishError } = useStockfishEvaluation(position ?? null, reviewFens);
  const moveReviews = useMemo(() => reviewedMoves.map(move => reviewMove(move, reviews.get(move.before), reviews.get(move.after))), [reviewedMoves, reviews]);
  const lastMove = reviewedMoves.at(-1);
  const checkedKing = useMemo(() => {
    if (!position) return null;
    const board = new Chess(position);
    if (!board.isCheck()) return null;
    return board.board().flat().find(piece => piece?.type === 'k' && piece.color === board.turn())?.square ?? null;
  }, [position]);
  const lastReview = moveReviews.at(-1);
  const roastContext = useMemo<RoastContext | null>(() => {
    if (!game || replay) return null;
    const history = game.board.history({ verbose: true });
    let humanIndex = -1, botIndex = -1;
    history.forEach((move, index) => { if (move.color === game.human) humanIndex = index; else botIndex = index; });
    return { ply: history.length, human: game.human, lastMove: history.at(-1) ?? null,
      humanMove: history[humanIndex] ?? null, humanGrade: moveReviews[humanIndex]?.grade,
      botGrade: moveReviews[botIndex]?.grade, result: game.result };
  }, [game, revision, replay, moveReviews]);
  const lastSquare = lastMove ? {
    left: `${(bottomColor === 'w' ? lastMove.to.charCodeAt(0) - 97 : 104 - lastMove.to.charCodeAt(0)) * 12.5}%`,
    top: `${(bottomColor === 'w' ? 8 - Number(lastMove.to[1]) : Number(lastMove.to[1]) - 1) * 12.5}%`,
  } : undefined;

  return <div className="tonkai-page">
    <header className="tk-header">
      <h1>{BOT_NAME}</h1>
      <div className="tk-header-actions">
      <div className="tk-tabs" role="group" aria-label="Vista do tabuleiro">
        <button aria-pressed={!replay} onClick={() => setView('play')}>Jogar</button>
        <button aria-pressed={replay} onClick={() => setView('history')}>Histórico</button>
      </div>
      <Link className="tk-versions-link" to="/tonkai/versoes">Versões · G16 v2 <span aria-hidden="true">↗</span></Link>
      </div>
    </header>
    {loadError && !replay && <p role="alert">{loadError} <button onClick={() => setAttempt(a => a + 1)}>Tentar novamente</button></p>}
    <div className="tk-layout">
      <section className="tk-board-panel" aria-label={replay ? 'Reprodução do jogo' : 'Jogar xadrez'}>
        <div className="tk-toolbar">
          <div className="tk-match-controls">
          {replay ? <>
            <span className="tk-status" role="status" aria-label="Estado da reprodução">{replayWin ? currentPly === 0 ? 'Posição inicial' : `${currentPly} / ${replayWin.plies} lances` : 'Histórico vazio'}</span>
            <span className="tk-readonly">Reprodução</span>
          </> : <>
            {(!game || game.over || playError) && <label className="tk-colour"><span className="tk-sr-only">A tua cor</span><select aria-label="A tua cor" value={colour} onChange={e => setColour(e.target.value as Color)}>
              <option value="w">Brancas</option><option value="b">Pretas</option>
            </select></label>}
            {(!game || game.over || playError) && <button className="tk-start" onClick={start} disabled={!network}>{game ? 'Nova partida' : network ? 'Começar' : 'A carregar…'}</button>}
          </>}
          </div>
          <div className="tk-tools-controls">
          {!replay && <span className="tk-time-control" title="5 minutos + 3 segundos por lance" aria-label="Ritmo: 5 minutos mais 3 segundos">5 + 3</span>}
          <button className="tk-evaluation-toggle" aria-pressed={showEvaluations}
            aria-label={showEvaluations ? 'Ocultar avaliações' : 'Mostrar avaliações'}
            title={showEvaluations ? 'Ocultar avaliações' : 'Mostrar avaliações'} onClick={() => setShowEvaluations(value => !value)}>
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M5 20V10M12 20V4M19 20v-7" strokeLinecap="round" /></svg>
            <span>Análise</span>
          </button>
          {!replay && game && !game.over && <button className="tk-resign" aria-label="Desistir" title="Desistir" onClick={() => {
            if (!game.checkClock()) { game.resigned = true; game.predictions = []; game.clock?.stop(); }
            setSelected(null); setPromotion(null); setRevision(n => n + 1);
          }}><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M5 21V3m0 1c5-4 9 4 14 0v10c-5 4-9-4-14 0" strokeLinecap="round" strokeLinejoin="round" /></svg><span>Desistir</span></button>}
          </div>
        </div>
        {playError && !replay && <p role="alert">{playError}</p>}
        <PlayerIdentity name={replay ? replayWin?.opponent ?? 'Adversário' : BOT_NAME}
          bot={!replay} color={topColor} active={playing && turn === topColor}
          clock={replay ? undefined : game?.clock ?? null} flagged={game?.timeoutColor === topColor} />
        <div className={`tk-board-with-evals${showEvaluations ? '' : ' tk-board-without-evals'}`}>
          {showEvaluations && <TonkaiEvalBars stockfish={stockfish} tonkinho={evaluation} bottomColor={bottomColor} error={stockfishError} />}
        <div className="tk-board" data-fen={position}>
          <Chessboard options={{ id: 'tonkai-board', position, pieces: TONKAI_PIECES,
            boardOrientation: bottomColor === 'w' ? 'white' : 'black',
            allowDragging: !replay && !!game?.humanTurn && !promotion && !playError, allowDrawingArrows: false,
            darkSquareStyle: { backgroundColor: '#789464' }, lightSquareStyle: { backgroundColor: '#eeeed2' },
            darkSquareNotationStyle: { color: '#f0f1dc', fontFamily: 'system-ui, sans-serif', fontWeight: 750, fontSize: 'clamp(10px, 1vw, 14px)' },
            lightSquareNotationStyle: { color: '#527247', fontFamily: 'system-ui, sans-serif', fontWeight: 750, fontSize: 'clamp(10px, 1vw, 14px)' },
            alphaNotationStyle: { bottom: 2, right: 4 }, numericNotationStyle: { top: 2, left: 4 },
            animationDurationInMs: 180,
            draggingPieceStyle: { filter: 'drop-shadow(0 5px 5px #14211755)' },
            draggingPieceGhostStyle: { opacity: .2 },
            dropSquareStyle: { boxShadow: 'inset 0 0 0 3px #fff9' },
            canDragPiece: ({ piece }) => !replay && piece.pieceType[0] === game?.human,
            squareRenderer: ({ square, children }) => <div className={[
              'tk-square',
              (lastMove?.from === square || lastMove?.to === square) && 'tk-square-last',
              activeSelection === square && 'tk-square-selected',
              checkedKing === square && 'tk-square-check',
            ].filter(Boolean).join(' ')}>
              {children}
              {destinations.has(square as Square) && <span aria-hidden="true"
                className={`tk-legal-move${destinations.get(square as Square) ? ' tk-legal-capture' : ''}`} />}
            </div>,
            onPieceDrag: ({ square }) => setSelected(square),
            onPieceDragCancel: () => setSelected(null),
            onPieceDrop: ({ sourceSquare, targetSquare }) => targetSquare ? tryMove(sourceSquare, targetSquare) : false,
            onSquareClick: ({ square }) => {
              if (replay || !game?.humanTurn || promotion || playError) return;
              if (selected && tryMove(selected, square)) return;
              setSelected(selected !== square && game.board.get(square as Square)?.color === game.human ? square : null);
            },
          }} />
          {!replay && showEvaluations && evaluation && <div className="tk-piece-values" aria-label="Valores posicionais das peças">
            {evaluation.pieces.map(piece => {
              const file = piece.square.charCodeAt(0) - 97, rank = Number(piece.square[1]) - 1;
              return <div key={piece.square} className="tk-piece-value-cell" style={{
                gridColumn: bottomColor === 'w' ? file + 1 : 8 - file,
                gridRow: bottomColor === 'w' ? 8 - rank : rank + 1,
              }}><span className={`tk-piece-value${piece.value !== null && piece.value < -.05 ? ' tk-piece-value-negative' : ''}`}
                data-value-square={piece.square} data-value={piece.value ?? 'king'}
                aria-label={`${piece.square}: ${piece.value === null ? 'Rei, sem valor de troca finito' : pieceValueLabel(piece.value)}`}>
                {pieceValueLabel(piece.value)}
              </span></div>;
            })}
          </div>}
          {lastReview && <div className="tk-move-marker" style={lastSquare}>
            <TonkaiMoveIcon grade={lastReview.grade} description={reviewDescription(lastReview)} />
          </div>}
          {promotion && !replay && <div className="tk-board-overlay">
            <div className="tk-board-dialog" role="dialog" aria-modal="true" aria-label="Escolher promoção">
              <span className="tk-dialog-eyebrow">Peão na última fila</span>
              <h2>Escolher promoção</h2>
              <p>Escolhe a peça para concluir a jogada.</p>
              <div className="tk-promotion-options">
                {([['q', 'Dama'], ['r', 'Torre'], ['b', 'Bispo'], ['n', 'Pónei']] as const).map(([p, label]) =>
                  <button key={p} onClick={() => commitMove(promotion.from + promotion.to + p)}><TonkaiPiece code={game!.human + p.toUpperCase()} />{label}</button>)}
              </div>
              <button className="tk-dialog-secondary" onClick={() => setPromotion(null)}>Cancelar</button>
            </div>
          </div>}
          {!replay && game?.over && !resultDismissed && <div className="tk-board-overlay">
            <div className="tk-board-dialog tk-result-dialog" role="dialog" aria-modal="true" aria-label="Resultado da partida">
              <span className="tk-dialog-eyebrow">Partida terminada</span>
              <h2>{game.result === 'draw' ? 'Empate' : game.result === 'loss' ? 'Ganhaste!' : `${BOT_NAME} venceu`}</h2>
              <p>{resultDetail}</p>
              <div className="tk-result-actions">
                <button onClick={start}>Jogar novamente</button>
                <button className="tk-dialog-secondary" onClick={() => setResultDismissed(true)}>Ver tabuleiro</button>
              </div>
            </div>
          </div>}
        </div>
        </div>
        <PlayerIdentity name={replay ? BOT_NAME : playerName} bot={replay}
          photo={replay ? undefined : auth?.user?.photoURL} color={bottomColor} active={playing && turn === bottomColor}
          clock={replay ? undefined : game?.clock ?? null} flagged={game?.timeoutColor === bottomColor} />
        {replay ? <div className="tk-replay-controls" role="group" aria-label="Navegar na partida">
          <button aria-label="Posição inicial" disabled={!currentPly} onClick={() => setReplayPly(0)}>⏮</button>
          <button aria-label="Lance anterior" disabled={!currentPly} onClick={() => setReplayPly(currentPly - 1)}>←</button>
          <button aria-label="Próximo lance" disabled={!replayWin || currentPly === replayWin.plies} onClick={() => setReplayPly(currentPly + 1)}>→</button>
          <button aria-label="Posição final" disabled={!replayWin || currentPly === replayWin.plies} onClick={() => setReplayPly(replayWin?.plies ?? 0)}>⏭</button>
        </div> : <p className="tk-status tk-live-status" role="status" aria-label="Estado do jogo"><span className={game?.humanTurn ? 'tk-dot tk-dot-active' : 'tk-dot'} />{status}</p>}
        {lastMove && <details className="tk-last-review">
          <summary>{lastReview ? <><TonkaiMoveIcon grade={lastReview.grade} /><strong>{lastMove.san}</strong> · {MOVE_GRADES[lastReview.grade].label}</>
            : <><strong>{lastMove.san}</strong> · {stockfishError ? 'Análise indisponível' : 'A analisar o lance…'}</>}</summary>
          {lastReview && <p>{reviewDescription(lastReview)}</p>}
          <p>Classificação estimada pelo Stockfish, comparando a vantagem antes e depois do lance. “Possível brilhante” exige um sacrifício forte e análise mais profunda. São critérios nossos, não os do Chess.com.</p>
        </details>}
      </section>
      <aside className="tk-sidebar">
        <TonkaiRoastChat key={gameId.current || 'before-game'}
          context={roastContext} canInterject={!replay && !!game && (game.over || game.humanTurn)} />
      </aside>
    </div>
  </div>;
}
