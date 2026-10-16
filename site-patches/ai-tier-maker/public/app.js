(() => {
  'use strict';

  const MODELS = window.TIER_MODELS;
  const PROVIDERS = window.TIER_PROVIDERS;
  const LOGOS = window.TIER_LOGOS;
  const BY_ID = Object.fromEntries(MODELS.map(m => [m.id, m]));
  const STORAGE_KEY = 'ai-tier-maker/v1';

  const DEFAULT_TIERS = [
    { label: 'S', color: '#ff7f7f' },
    { label: 'A', color: '#ffbf7f' },
    { label: 'B', color: '#ffdf7f' },
    { label: 'C', color: '#ffff7f' },
    { label: 'D', color: '#bfff7f' },
    { label: 'F', color: '#7fbfff' },
  ];

  const boardEl = document.getElementById('board');
  const poolEl = document.getElementById('pool');
  const poolCountEl = document.getElementById('pool-count');
  const toastEl = document.getElementById('toast');

  let state = load() || freshState();
  let selectedId = null;
  const cardEls = new Map();

  // ---------------------------------------------------------------- state

  function freshState() {
    return {
      tiers: DEFAULT_TIERS.map((t, i) => ({ id: 't' + Date.now().toString(36) + i, label: t.label, color: t.color, items: [] })),
      pool: MODELS.map(m => m.id),
      showRank: true,
    };
  }

  function load() {
    try {
      const s = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (!s || !Array.isArray(s.tiers) || !Array.isArray(s.pool)) return null;
      // Garante que todos os modelos existem exatamente uma vez (o data.js pode ter mudado).
      const seen = new Set();
      const keep = id => BY_ID[id] && !seen.has(id) && seen.add(id);
      s.tiers.forEach(t => { t.items = t.items.filter(keep); });
      s.pool = s.pool.filter(keep);
      MODELS.forEach(m => { if (!seen.has(m.id)) s.pool.push(m.id); });
      return s;
    } catch { return null; }
  }

  function save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }

  function syncFromDom() {
    const ids = zone => [...zone.querySelectorAll('.card')].map(c => c.dataset.id);
    state.tiers.forEach(t => {
      const zone = boardEl.querySelector(`.dropzone[data-zone="${t.id}"]`);
      if (zone) t.items = ids(zone);
    });
    state.pool = ids(poolEl);
    updateCount();
    save();
  }

  // ---------------------------------------------------------------- render

  function cardFor(id) {
    if (cardEls.has(id)) return cardEls.get(id);
    const m = BY_ID[id];
    const p = PROVIDERS[m.provider];
    const el = document.createElement('div');
    el.className = 'card';
    el.dataset.id = id;
    el.tabIndex = 0;
    el.style.setProperty('--pc', p.color);
    el.title = `${m.name}${m.effort ? ` (${m.effort})` : ''}\n${p.name} · ${m.license}\n#${m.rank} · +${m.score.toFixed(2)}% ±${m.ci.toFixed(2)}%`;
    el.setAttribute('aria-label', `${m.name} ${m.effort}, ${p.name}, posição ${m.rank}`);
    el.innerHTML = `
      <span class="rank">#${m.rank}</span>
      <span class="score">+${m.score.toFixed(1)}</span>
      <img src="${LOGOS[m.provider]}" alt="${p.name}" draggable="false">
      <span class="name"></span>
      <span class="effort"></span>`;
    el.querySelector('.name').textContent = m.name;
    el.querySelector('.effort').textContent = m.effort;
    cardEls.set(id, el);
    return el;
  }

  function render() {
    boardEl.innerHTML = '';
    state.tiers.forEach((t, i) => boardEl.appendChild(tierRow(t, i)));
    poolEl.replaceChildren(...state.pool.map(cardFor));
    document.body.classList.toggle('hide-rank', !state.showRank);
    document.getElementById('toggle-rank').checked = state.showRank;
    markSelected();
    updateCount();
  }

  function tierRow(t, index) {
    const row = document.createElement('div');
    row.className = 'tier';
    row.style.setProperty('--tier-color', t.color);
    row.style.animationDelay = `${index * 40}ms`;

    const label = document.createElement('div');
    label.className = 'tier-label';
    const text = document.createElement('span');
    text.contentEditable = 'true';
    text.spellcheck = false;
    text.textContent = t.label;
    text.setAttribute('aria-label', 'Nome do tier');
    text.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); text.blur(); } });
    text.addEventListener('blur', () => {
      t.label = text.textContent.trim() || '?';
      text.textContent = t.label;
      save();
    });
    label.appendChild(text);

    const zone = document.createElement('div');
    zone.className = 'dropzone';
    zone.dataset.zone = t.id;
    zone.append(...t.items.map(cardFor));

    const tools = document.createElement('div');
    tools.className = 'tier-tools';
    tools.append(
      iconBtn('▲', 'Subir tier', () => moveTier(t, -1)),
      colorBtn(t, row),
      iconBtn('▼', 'Descer tier', () => moveTier(t, 1)),
      iconBtn('⟲', 'Devolver modelos à lista', () => clearTier(t)),
      iconBtn('✕', 'Apagar tier', () => deleteTier(t)),
    );

    row.append(label, zone, tools);
    return row;
  }

  function iconBtn(glyph, title, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'icon-btn';
    b.title = title;
    b.setAttribute('aria-label', title);
    b.textContent = glyph;
    b.addEventListener('click', onClick);
    return b;
  }

  function colorBtn(t, row) {
    const b = document.createElement('label');
    b.className = 'icon-btn';
    b.title = 'Cor do tier';
    b.innerHTML = '<span class="swatch"></span>';
    const input = document.createElement('input');
    input.type = 'color';
    input.value = t.color;
    input.setAttribute('aria-label', 'Cor do tier');
    input.addEventListener('input', () => { t.color = input.value; row.style.setProperty('--tier-color', t.color); });
    input.addEventListener('change', save);
    b.appendChild(input);
    return b;
  }

  function updateCount() {
    poolCountEl.textContent = `${state.pool.length}/${MODELS.length}`;
  }

  // ---------------------------------------------------------------- tier ops

  function moveTier(t, dir) {
    syncFromDom();
    const i = state.tiers.indexOf(t);
    const j = i + dir;
    if (j < 0 || j >= state.tiers.length) return;
    [state.tiers[i], state.tiers[j]] = [state.tiers[j], state.tiers[i]];
    save();
    render();
  }

  function clearTier(t) {
    syncFromDom();
    state.pool.push(...t.items);
    t.items = [];
    save();
    render();
  }

  function deleteTier(t) {
    syncFromDom();
    if (state.tiers.length <= 1) return toast('Tem de ficar pelo menos um tier');
    state.pool.push(...t.items);
    state.tiers = state.tiers.filter(x => x !== t);
    save();
    render();
    toast(`Tier “${t.label}” apagado`);
  }

  function addTier() {
    syncFromDom();
    const hue = Math.floor(Math.random() * 360);
    state.tiers.push({ id: 't' + Date.now().toString(36), label: 'Novo', color: hslToHex(hue, 85, 75), items: [] });
    save();
    render();
  }

  function autoArena() {
    // Cortes nos "saltos" naturais da pontuação da Arena.
    const cuts = [13.5, 9, 6, 3.7, 3.0];
    state.tiers.forEach(t => { t.items = []; });
    state.pool = [];
    MODELS.forEach(m => {
      let idx = cuts.findIndex(c => m.score >= c);
      if (idx === -1) idx = cuts.length;
      state.tiers[Math.min(idx, state.tiers.length - 1)].items.push(m.id);
    });
    save();
    render();
    toast('Distribuído pela pontuação da Arena');
  }

  function reset() {
    const showRank = state.showRank;
    state = freshState();
    state.showRank = showRank;
    selectedId = null;
    save();
    render();
    toast('Tier list reposta');
  }

  // ---------------------------------------------------------------- selection + keyboard

  function markSelected() {
    cardEls.forEach((el, id) => el.classList.toggle('selected', id === selectedId));
  }

  function select(id) {
    selectedId = selectedId === id ? null : id;
    markSelected();
  }

  function sendSelectedTo(zone) {
    const el = cardEls.get(selectedId);
    if (!el || !zone) return;
    flip([el.parentElement, zone], () => zone.appendChild(el));
    syncFromDom();
  }

  document.addEventListener('keydown', e => {
    if (e.target.isContentEditable || e.target.tagName === 'INPUT') return;
    if (e.key === 'Enter' && e.target.classList?.contains('card')) {
      select(e.target.dataset.id);
      return;
    }
    if (!selectedId || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Escape') {
      sendSelectedTo(poolEl);
      selectedId = null;
      markSelected();
      return;
    }
    const k = e.key.toLowerCase();
    let tier = state.tiers.find(t => t.label.toLowerCase().startsWith(k));
    if (!tier && /^[1-9]$/.test(k)) tier = state.tiers[Number(k) - 1];
    if (tier) {
      e.preventDefault();
      sendSelectedTo(boardEl.querySelector(`.dropzone[data-zone="${tier.id}"]`));
    }
  });

  // ---------------------------------------------------------------- FLIP animation

  function flip(zones, mutate) {
    const cards = [...new Set(zones.filter(Boolean))].flatMap(z => [...z.querySelectorAll('.card')]);
    const before = new Map(cards.map(c => [c, c.getBoundingClientRect()]));
    mutate();
    before.forEach((r, c) => {
      if (c.classList.contains('dragging-src')) return;
      const n = c.getBoundingClientRect();
      const dx = r.left - n.left, dy = r.top - n.top;
      if (dx || dy) {
        c.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], { duration: 200, easing: 'cubic-bezier(.2,.8,.2,1)' });
      }
    });
  }

  // ---------------------------------------------------------------- pointer drag & drop

  let drag = null;

  document.addEventListener('pointerdown', e => {
    const card = e.target.closest('.card');
    if (!card || e.button !== 0) return;
    const r = card.getBoundingClientRect();
    drag = { card, startX: e.clientX, startY: e.clientY, offX: e.clientX - r.left, offY: e.clientY - r.top, active: false, ghost: null, overZone: null, lastX: e.clientX, lastY: e.clientY };
    card.setPointerCapture?.(e.pointerId);
  });

  document.addEventListener('pointermove', e => {
    if (!drag) return;
    drag.lastX = e.clientX; drag.lastY = e.clientY;
    if (!drag.active) {
      if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) < 5) return;
      startDrag();
    }
    moveGhost(e.clientX, e.clientY);
    placeAt(e.clientX, e.clientY);
  });

  document.addEventListener('pointerup', endPointer);
  document.addEventListener('pointercancel', endPointer);

  function startDrag() {
    drag.active = true;
    const g = drag.card.cloneNode(true);
    g.classList.remove('selected');
    g.classList.add('ghost');
    g.removeAttribute('tabindex');
    document.body.appendChild(g);
    drag.ghost = g;
    drag.card.classList.add('dragging-src');
    document.body.style.cursor = 'grabbing';
    autoScroll();
  }

  function moveGhost(x, y) {
    drag.ghost.style.transform = `translate(${x - drag.offX}px, ${y - drag.offY}px)`;
  }

  function placeAt(x, y) {
    const zone = document.elementFromPoint(x, y)?.closest('.dropzone');
    if (drag.overZone !== zone) {
      drag.overZone?.classList.remove('is-over');
      zone?.classList.add('is-over');
      drag.overZone = zone;
    }
    if (!zone) return;

    const el = drag.card;
    // Offsets de layout em vez de getBoundingClientRect: ignoram as transformações das
    // animações FLIP, senão os cards ficavam a oscilar entre duas posições.
    const zr = zone.getBoundingClientRect();
    const ox = zr.left + zone.clientLeft, oy = zr.top + zone.clientTop;
    let ref = null;
    for (const c of zone.querySelectorAll('.card')) {
      if (c === el) continue;
      const left = ox + c.offsetLeft, top = oy + c.offsetTop;
      if (y < top || (y <= top + c.offsetHeight && x < left + c.offsetWidth / 2)) { ref = c; break; }
    }
    if (el.parentElement === zone && (ref === el.nextElementSibling || (ref === null && el === zone.lastElementChild))) return;
    flip([el.parentElement, zone], () => zone.insertBefore(el, ref));
  }

  function autoScroll() {
    if (!drag || !drag.active) return;
    const edge = 70, y = drag.lastY, h = window.innerHeight;
    if (y < edge) window.scrollBy(0, -Math.ceil((edge - y) / 4));
    else if (y > h - edge) window.scrollBy(0, Math.ceil((y - (h - edge)) / 4));
    requestAnimationFrame(autoScroll);
  }

  function endPointer() {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (!d.active) {
      select(d.card.dataset.id);
      return;
    }
    d.overZone?.classList.remove('is-over');
    d.card.classList.remove('dragging-src');
    document.body.style.cursor = '';
    // Anima o ghost a "aterrar" no sítio final.
    const target = d.card.getBoundingClientRect();
    const anim = d.ghost.animate(
      [
        { transform: d.ghost.style.transform, rotate: '-3deg', scale: '1.06' },
        { transform: `translate(${target.left}px, ${target.top}px)`, rotate: '0deg', scale: '1' },
      ],
      { duration: 160, easing: 'ease-out', fill: 'forwards' });
    anim.onfinish = () => d.ghost.remove();
    syncFromDom();
  }

  // ---------------------------------------------------------------- PNG export

  async function exportPng() {
    syncFromDom();
    await document.fonts.ready;

    const logoImgs = {};
    await Promise.all(Object.entries(LOGOS).map(([k, src]) => new Promise(res => {
      const img = new Image();
      img.onload = img.onerror = () => res();
      img.src = src;
      logoImgs[k] = img;
    })));

    const S = 2, W = 1600, PAD = 40, HEAD = 110, LABEL_W = 150, CW = 150, CH = 124, GAP = 12, FOOT = 50;
    const areaW = W - PAD * 2 - LABEL_W;
    const perRow = Math.max(1, Math.floor((areaW - GAP) / (CW + GAP)));
    const rowHeights = state.tiers.map(t => {
      const lines = Math.max(1, Math.ceil(t.items.length / perRow));
      return lines * (CH + GAP) + GAP;
    });
    const H = HEAD + rowHeights.reduce((a, b) => a + b, 0) + FOOT + PAD;

    const cv = document.createElement('canvas');
    cv.width = W * S; cv.height = H * S;
    const ctx = cv.getContext('2d');
    ctx.scale(S, S);

    // Fundo
    ctx.fillStyle = '#0b0d12';
    ctx.fillRect(0, 0, W, H);
    const glow = (x, y, r, c) => { const g = ctx.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, c); g.addColorStop(1, 'transparent'); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H); };
    glow(W * 0.12, 40, 600, 'rgba(124, 77, 255, 0.22)');
    glow(W * 0.88, 80, 520, 'rgba(0, 190, 230, 0.14)');

    // Título
    const grad = ctx.createLinearGradient(PAD, 0, PAD + 420, 0);
    grad.addColorStop(0, '#ffffff'); grad.addColorStop(0.6, '#a07bff'); grad.addColorStop(1, '#2ec9f0');
    ctx.fillStyle = grad;
    ctx.font = '700 40px "Space Grotesk", sans-serif';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('AI Tier List', PAD, PAD + 34);
    ctx.fillStyle = '#8b91a1';
    ctx.font = '400 17px "Space Grotesk", sans-serif';
    ctx.fillText('Top 20 · LMArena Text · Geral', PAD, PAD + 62);

    // Tiers
    let y = HEAD;
    const boardX = PAD, boardW = W - PAD * 2;
    ctx.save();
    roundRect(ctx, boardX, y, boardW, rowHeights.reduce((a, b) => a + b, 0), 16);
    ctx.clip();
    ctx.fillStyle = '#12151c';
    ctx.fillRect(boardX, y, boardW, H);

    state.tiers.forEach((t, i) => {
      const rh = rowHeights[i];
      ctx.fillStyle = t.color;
      ctx.fillRect(boardX, y, LABEL_W, rh);
      ctx.fillStyle = '#111';
      fitText(ctx, t.label, boardX + LABEL_W / 2, y + rh / 2, LABEL_W - 20, 40, '700', '"Space Grotesk", sans-serif');

      t.items.forEach((id, k) => {
        const cx = boardX + LABEL_W + GAP + (k % perRow) * (CW + GAP);
        const cy = y + GAP + Math.floor(k / perRow) * (CH + GAP);
        drawCard(ctx, BY_ID[id], cx, cy, CW, CH, logoImgs);
      });

      y += rh;
      if (i < state.tiers.length - 1) {
        ctx.fillStyle = 'rgba(255,255,255,0.07)';
        ctx.fillRect(boardX, y - 1, boardW, 1);
      }
    });
    ctx.restore();

    ctx.fillStyle = '#5d6373';
    ctx.font = '400 13px "Space Grotesk", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('feito com AI Tier Maker · dados LMArena', W - PAD, H - PAD + 14);

    cv.toBlob(blob => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'ai-tier-list.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      toast('PNG exportado');
    }, 'image/png');
  }

  function drawCard(ctx, m, x, y, w, h, logos) {
    const p = PROVIDERS[m.provider];
    ctx.save();
    roundRect(ctx, x, y, w, h, 12);
    ctx.fillStyle = '#1a1d26';
    ctx.fill();
    ctx.clip();
    const g = ctx.createRadialGradient(x + w / 2, y, 0, x + w / 2, y, h);
    g.addColorStop(0, hexA(p.color, 0.3)); g.addColorStop(1, 'transparent');
    ctx.fillStyle = g;
    ctx.fillRect(x, y, w, h);
    ctx.restore();

    ctx.save();
    roundRect(ctx, x + 0.5, y + 0.5, w - 1, h - 1, 12);
    ctx.strokeStyle = hexA(p.color, 0.5);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    const img = logos[m.provider];
    if (img && img.complete) ctx.drawImage(img, x + w / 2 - 19, y + 14, 38, 38);

    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#eef0f5';
    ctx.font = '700 13px "JetBrains Mono", monospace';
    const lines = wrap(ctx, m.name, w - 16).slice(0, 2);
    const nameY = y + 74 + (lines.length === 1 ? 6 : 0);
    lines.forEach((l, i) => ctx.fillText(l, x + w / 2, nameY + i * 16));

    if (m.effort) {
      ctx.fillStyle = '#8b91a1';
      ctx.font = '500 10px "JetBrains Mono", monospace';
      ctx.fillText(m.effort.toUpperCase(), x + w / 2, y + h - 10);
    }

    if (state.showRank) {
      ctx.font = '700 10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillStyle = '#8b91a1';
      ctx.fillText('#' + m.rank, x + 8, y + 16);
      ctx.textAlign = 'right';
      ctx.fillStyle = '#5fd283';
      ctx.fillText('+' + m.score.toFixed(1), x + w - 8, y + 16);
    }
  }

  function wrap(ctx, text, maxW) {
    const words = text.split(' ');
    const lines = [];
    let line = '';
    for (const w of words) {
      const test = line ? line + ' ' + w : w;
      if (ctx.measureText(test).width > maxW && line) { lines.push(line); line = w; }
      else line = test;
    }
    if (line) lines.push(line);
    return lines;
  }

  function fitText(ctx, text, cx, cy, maxW, size, weight, family) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    let s = size;
    do { ctx.font = `${weight} ${s}px ${family}`; s -= 2; } while (ctx.measureText(text).width > maxW && s > 10);
    ctx.fillText(text, cx, cy);
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }

  function hslToHex(h, s, l) {
    s /= 100; l /= 100;
    const k = n => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = n => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1)))).toString(16).padStart(2, '0');
    return `#${f(0)}${f(8)}${f(4)}`;
  }

  // ---------------------------------------------------------------- misc

  let toastTimer;
  function toast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove('show'), 1800);
  }

  document.getElementById('btn-add-tier').addEventListener('click', addTier);
  document.getElementById('btn-auto').addEventListener('click', autoArena);
  document.getElementById('btn-reset').addEventListener('click', reset);
  document.getElementById('btn-export').addEventListener('click', exportPng);
  document.getElementById('toggle-rank').addEventListener('change', e => {
    state.showRank = e.target.checked;
    document.body.classList.toggle('hide-rank', !state.showRank);
    save();
  });

  render();
})();
