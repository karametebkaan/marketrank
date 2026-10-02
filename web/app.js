/* global deck */
'use strict';
const Q = new URLSearchParams(location.search);
const S = {
  status: null, times: [], frame: null, raster: null, shock: null, shockRaster: null, shockScale: 1, shockVmax: 1,
  heightBase: 1, deck: null, playing: false, loadedKey: '', loadedGen: null, selftest: Q.has('selftest'),
  frameSeq: 0, shockSeq: 0, idx: null, idxKey: '', colCache: null, arcsInit: false, selftestDone: false,
};
const $ = (id) => document.getElementById(id);
const MID = [247, 247, 247], POS = [178, 24, 43], NEG = [33, 102, 172];

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.text();
  if (!r.ok) throw new Error(`${url}: ${r.status} ${body}`);
  return JSON.parse(body);
}
async function getFloat32(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}

function span(L) { return Math.max(L.cols, L.rows); }

// Triangle topology of a w×h vertex grid is fixed: build it once per (w, h).
function gridIndices(w, h) {
  const key = `${w}x${h}`;
  if (S.idxKey === key) return S.idx;
  const idx = new Uint32Array(Math.max(0, (w - 1) * (h - 1) * 6));
  let p = 0;
  for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
    const a = y * w + x, b = a + 1, c = a + w, d = c + 1;
    idx[p++] = a; idx[p++] = b; idx[p++] = d; idx[p++] = a; idx[p++] = d; idx[p++] = c;
  }
  S.idx = idx; S.idxKey = key;
  return idx;
}

// Per-vertex diverging colours (float RGB 0..1) by z / vmax: red > 0, blue < 0, near-white at 0.
// Cached per (raster, vmax), so height-scale changes reuse them.
function vertexColors(z, vmax) {
  const c = S.colCache;
  if (c && c.z === z && c.vmax === vmax) return c.col;
  const n = z.length, col = new Float32Array(n * 3), inv = vmax > 0 ? 1 / vmax : 0;
  for (let k = 0; k < n; k++) {
    let x = z[k] * inv;
    x = x > 1 ? 1 : (x < -1 ? -1 : x);
    const a = x < 0 ? -x : x, e = x < 0 ? NEG : POS;
    col[3 * k] = (MID[0] + (e[0] - MID[0]) * a) / 255;
    col[3 * k + 1] = (MID[1] + (e[1] - MID[1]) * a) / 255;
    col[3 * k + 2] = (MID[2] + (e[2] - MID[2]) * a) / 255;
  }
  S.colCache = { z, vmax, col };
  return col;
}

// Triangulated surface: one vertex per raster sample, Gouraud-coloured by z (per-vertex COLOR_0, consumed by
// SimpleMeshLayer as the `colors` attribute; no texture). Heights are z·scale clamped to ±clamp.
function buildTerrain(z, meta, lattice, scale, vmax, clamp) {
  const { w, h } = meta, sx = lattice.cols / w, sy = lattice.rows / h;
  const pos = new Float32Array(w * h * 3), nrm = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = y * w + x, v = z[k] * scale;
    pos[3 * k] = (x + 0.5) * sx; pos[3 * k + 1] = (y + 0.5) * sy; pos[3 * k + 2] = v > clamp ? clamp : (v < -clamp ? -clamp : v);
  }
  const zz = (k) => pos[3 * k + 2];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = y * w + x;
    const dx = (zz(y * w + Math.min(w - 1, x + 1)) - zz(y * w + Math.max(0, x - 1))) / (2 * sx);
    const dy = (zz(Math.min(h - 1, y + 1) * w + x) - zz(Math.max(0, y - 1) * w + x)) / (2 * sy);
    const len = Math.hypot(dx, dy, 1);
    nrm[3 * k] = -dx / len; nrm[3 * k + 1] = -dy / len; nrm[3 * k + 2] = 1 / len;
  }
  return {
    topology: 'triangle-list',
    attributes: { POSITION: { value: pos, size: 3 }, NORMAL: { value: nrm, size: 3 }, COLOR_0: { value: vertexColors(z, vmax), size: 3 } },
    indices: { value: gridIndices(w, h), size: 1 },
  };
}

// Raster vertex nearest the centre of lattice cell (col, row): its position (lattice units) and value.
function sample(z, meta, lattice, col, row) {
  const px = Math.min(meta.w - 1, Math.floor((col + 0.5) * meta.w / lattice.cols));
  const py = Math.min(meta.h - 1, Math.floor((row + 0.5) * meta.h / lattice.rows));
  return { x: (px + 0.5) * lattice.cols / meta.w, y: (py + 0.5) * lattice.rows / meta.h, v: z[py * meta.w + px] };
}

function signColor(v, alpha) {
  if (v === null || v === undefined || !Number.isFinite(v) || v === 0) return [150, 150, 150, alpha];
  return v > 0 ? [...POS, alpha] : [...NEG, alpha];
}

function render() {
  const f = S.frame;
  if (!f || !S.deck) return;
  const L = f.lattice, shock = !!S.shock;
  const meta = shock ? S.shock.raster : f.raster, z = shock ? S.shockRaster : S.raster;
  const hs = parseFloat($('hscale').value);
  const scale = (shock ? S.shockScale : S.heightBase) * hs;
  const vmax = shock ? S.shockVmax : Math.max(Math.abs(meta.zmin), Math.abs(meta.zmax), 1e-9);
  const clamp = shock ? 0.35 * span(L) * hs : Infinity;
  const nodes = f.nodes.map((n) => {
    const col = n[3] % L.cols, row = Math.floor(n[3] / L.cols), s = sample(z, meta, L, col, row);
    const zt = Math.max(-clamp, Math.min(clamp, s.v * scale));
    return { i: n[0], ticker: n[1], sector: n[2], x: s.x, y: s.y, zt, v: s.v, h: n[6], pi: n[8], score: n[9] };
  });
  const byI = new Map(nodes.map((n) => [n.i, n]));
  const pos = (n, lift = 0.3) => [n.x, n.y, n.zt + lift];
  const ranked = nodes.filter((n) => n.h !== null).sort((a, b) => b.h - a.h);
  const labeled = ranked.slice(0, 30).concat(ranked.slice(-30));
  const nArcs = shock ? 0 : Number($('arcs').value);
  const arcs = f.arcs.slice(0, nArcs).map((a) => ({ s: byI.get(a[0]), t: byI.get(a[1]), w: a[2] })).filter((a) => a.s && a.t);
  const wmax = arcs.reduce((m, a) => Math.max(m, a.w), 1e-12);
  const holdings = (f.portfolio || []).filter((p) => p.i !== null && byI.has(p.i)).map((p) => ({ ...byI.get(p.i), weight: p.weight }));
  const layers = [
    new deck.SimpleMeshLayer({
      id: 'terrain', data: [{}], mesh: buildTerrain(z, meta, L, scale, vmax, clamp),
      getPosition: [0, 0, 0], getColor: [255, 255, 255], material: { ambient: 0.55, diffuse: 0.55, shininess: 12, specularColor: [30, 30, 30] },
    }),
    new deck.ScatterplotLayer({
      id: 'nodes', data: nodes, getPosition: (n) => pos(n), getRadius: 0.16, radiusUnits: 'common',
      getFillColor: (n) => signColor(shock ? n.v : n.h, 220), pickable: true,
    }),
    new deck.ArcLayer({
      id: 'arcs', data: arcs, getSourcePosition: (a) => pos(a.s), getTargetPosition: (a) => pos(a.t),
      getWidth: (a) => 0.5 + 1.5 * a.w / wmax, widthUnits: 'pixels', getSourceColor: [255, 140, 0, 80], getTargetColor: [255, 215, 0, 80],
    }),
    new deck.ScatterplotLayer({
      id: 'portfolio', data: holdings, getPosition: (n) => pos(n, 0.4), getRadius: (n) => 0.35 + 1.5 * n.weight, radiusUnits: 'common',
      stroked: true, filled: false, getLineColor: [0, 150, 80], getLineWidth: 3, lineWidthUnits: 'pixels',
    }),
  ];
  if ($('labels').checked) {
    layers.push(new deck.TextLayer({
      id: 'labels', data: labeled, getPosition: (n) => pos(n, 0.9), getText: (n) => n.ticker, getSize: 12,
      getColor: [25, 25, 25], getTextAnchor: 'middle', getAlignmentBaseline: 'bottom', billboard: true,
    }));
  }
  S.deck.setProps({ layers });
}

function initDeck(lattice) {
  const el = $('canvas');
  S.deck = new deck.Deck({
    parent: el,
    views: new deck.OrbitView({ orbitAxis: 'Z', fovy: 40 }),
    initialViewState: { target: [lattice.cols / 2, lattice.rows / 2, 0], rotationX: 45, rotationOrbit: -25, zoom: Math.log2(Math.min(el.clientWidth, el.clientHeight) / (1.6 * span(lattice))), minZoom: -6, maxZoom: 12 },
    controller: true,
    getTooltip: ({ object }) => (object && object.ticker
      ? `${object.ticker} · ${object.sector}\nh ${object.h === null ? 'n/a' : object.h.toFixed(3)}  π ${object.pi === null ? 'n/a' : object.pi.toExponential(2)}\nscore+1 ${object.score === null ? 'n/a' : object.score.toFixed(3)}`
      : null),
  });
}

function setArcsLabel() { $('arcsLabel').textContent = $('arcs').value; }

// Server strings only ever reach the DOM through .value / textContent.
function fillTickers(nodes) {
  $('tickers').replaceChildren(...nodes.map((n) => { const o = document.createElement('option'); o.value = n[1]; return o; }));
}

function fillHoldings(f) {
  const byI = new Map(f.nodes.map((n) => [n[0], n]));
  const box = $('holdings');
  box.replaceChildren();
  (f.portfolio || []).forEach((p, k) => {
    const n = p.i === null ? undefined : byI.get(p.i);
    const h = n && n[6] !== null ? n[6] : null;
    if (k) box.appendChild(document.createTextNode('\n'));
    box.appendChild(document.createTextNode(`${p.ticker.padEnd(6)} ${(100 * p.weight).toFixed(1).padStart(5)}%  `));
    const el = document.createElement('span');
    if (h !== null) el.className = h >= 0 ? 'pos' : 'neg';
    el.textContent = h === null ? 'n/a' : h.toFixed(3);
    box.appendChild(el);
  });
}

// The server computes shocks only at its latest bar, so shocks are offered only while that bar is displayed.
function isLatestFrame() {
  return !!S.frame && S.times.length > 0 && S.frame.t === S.times[S.times.length - 1];
}
function updateShockEnabled() {
  const ok = isLatestFrame();
  $('shockApply').disabled = !ok;
  $('shockHint').hidden = ok;
}

function clearShock() {
  S.shock = null; S.shockRaster = null; S.shockSeq++;
  $('shockOut').textContent = '';
  showStatus();
}

// Loads frame t (latest when undefined). A response superseded by a newer request is dropped (returns false).
async function loadFrame(t) {
  const seq = ++S.frameSeq;
  const q = t === undefined || t === null ? '' : `?t=${t}`;
  const [f, z] = await Promise.all([getJSON(`/api/frame${q}`), getFloat32(`/api/frame/grid${q}`)]);
  if (seq !== S.frameSeq) return false;
  if (S.shock) clearShock();
  S.frame = f; S.raster = z;
  if (!S.deck) {
    const zmax = Math.max(Math.abs(f.raster.zmin), Math.abs(f.raster.zmax), 1e-9);
    S.heightBase = 0.2 * span(f.lattice) / zmax;
    initDeck(f.lattice);
    fillTickers(f.nodes);
  }
  if (!S.arcsInit) { S.arcsInit = true; $('arcs').value = Math.min(400, 2 * f.nodes.length); setArcsLabel(); }
  $('tlabel').textContent = `${f.time}  ·  ${f.nodes.length} active stocks  ·  ${f.params}`;
  fillHoldings(f);
  updateShockEnabled();
  render();
  return true;
}

async function refreshTimes() {
  S.times = await getJSON('/api/times');
  const s = $('scrub');
  s.max = Math.max(0, S.times.length - 1);
  if ($('follow').checked) s.value = s.max;
  updateShockEnabled();
}

function showStatus() {
  const st = S.status;
  if (!st) return;
  const tag = S.shock ? `  ·  Δh ${S.shock.shocked.map((x) => x.ticker).join(',')}` : '';
  $('status').textContent = st.error ? `error: ${st.error}` : `${st.label} · ${st.ready ? 'ready' : `computing ${st.computed}/${st.total}`} · ${st.nodes} stocks${tag}`;
}

async function applyShock() {
  if (!isLatestFrame()) throw new Error('shock applies to the latest bar');
  const seq = ++S.shockSeq;
  const tickers = [$('shockTicker').value.trim().toUpperCase()];
  const body = { shocks: tickers.map((ticker) => ({ ticker, size: Number($('shockSize').value) })) };
  const r = await getJSON('/api/shock', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const z = await getFloat32('/api/shock/grid');
  if (seq !== S.shockSeq) return;
  // Height and colour scale P: 95th percentile of |Δh| over active, non-shocked stocks (the shocked stock saturates).
  const L = S.frame.lattice, shocked = new Set(tickers);
  const mags = S.frame.nodes.filter((n) => !shocked.has(n[1]))
    .map((n) => Math.abs(sample(z, r.raster, L, n[3] % L.cols, Math.floor(n[3] / L.cols)).v))
    .filter(Number.isFinite).sort((a, b) => a - b);
  const p95 = Math.max(mags.length ? mags[Math.min(mags.length - 1, Math.floor(0.95 * mags.length))] : 0, 1e-9);
  S.shockVmax = p95; S.shockScale = 0.2 * span(L) / p95;
  S.shockRaster = z; S.shock = r;
  const fmt = (x) => `${x.ticker.padEnd(6)} ${x.dh >= 0 ? '+' : ''}${x.dh.toFixed(4)}`;
  $('shockOut').textContent = `Δh landscape · L1 Δπ ${r.l1_dpi.toExponential(2)}\n` +
    `shocked: ${r.shocked.map(fmt).join(', ')}\n\nreceivers\n${r.receivers.slice(0, 10).map(fmt).join('\n')}\n\nlosers\n${r.losers.slice(0, 10).map(fmt).join('\n')}`;
  showStatus();
  render();
}

async function onStatus(st) {
  S.status = st;
  showStatus();
  if (!st.ready) return;
  // Reload on a new generation, or (when following the latest bar) when new bars have been computed.
  const key = `${st.generation}:${st.computed}:${st.total}`;
  if (key === S.loadedKey) return;
  const newGen = st.generation !== S.loadedGen;
  if (newGen) clearShock();
  await refreshTimes();
  if (newGen || $('follow').checked) {
    if (!(await loadFrame(S.times[Number($('scrub').value)]))) return;
  }
  S.loadedKey = key; S.loadedGen = st.generation;
  if (S.selftest && !S.selftestDone) {
    S.selftestDone = true;
    // Optional ?shock=TICKER:SIZE exercises the shock view in the self-test (TICKER '*' = first active node).
    if (Q.has('shock')) {
      const [tk, sz] = Q.get('shock').split(':');
      $('shockTicker').value = tk === '*' ? S.frame.nodes[0][1] : tk;
      if (sz !== undefined) $('shockSize').value = sz;
      $('shockSizeLabel').textContent = `${$('shockSize').value}%`;
      await applyShock();
    }
    setTimeout(() => { document.title = `fluxscape-ok:${S.frame.nodes.length}`; }, 1500);
  }
}

function stopPlay() { S.playing = false; $('play').textContent = 'Play'; }

// Each tick waits for the previous frame load; a failure stops playback with a single error.
async function playTick() {
  if (!S.playing) return;
  const s = $('scrub');
  s.value = (Number(s.value) + 1) % (Number(s.max) + 1);
  try {
    await loadFrame(S.times[Number(s.value)]);
  } catch (e) { stopPlay(); fail(e); return; }
  if (S.playing) setTimeout(playTick, 700);
}

function wire() {
  // Scrubbing exits shock mode (as Reset does): the shock belongs to the latest bar only.
  $('scrub').addEventListener('input', () => { $('follow').checked = false; clearShock(); render(); $('shockApply').disabled = true; loadFrame(S.times[Number($('scrub').value)]).catch(fail); });
  $('play').addEventListener('click', () => {
    if (S.playing) { stopPlay(); return; }
    S.playing = true; $('play').textContent = 'Pause'; $('follow').checked = false;
    setTimeout(playTick, 0);
  });
  ['hscale', 'labels'].forEach((id) => $(id).addEventListener('input', render));
  $('arcs').addEventListener('input', () => { setArcsLabel(); render(); });
  $('apply').addEventListener('click', async () => {
    const body = { preset: $('preset').value, height: $('height').value, idw_power: Number($('idwPower').value), idw_radius: Number($('idwRadius').value), subdivision: Number($('subdiv').value) };
    if ($('href').value) body.h_ref = $('href').value;
    try { await getJSON('/api/params', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); S.deck && S.deck.finalize(); S.deck = null; } catch (e) { fail(e); }
  });
  $('shockSize').addEventListener('input', () => { $('shockSizeLabel').textContent = `${$('shockSize').value}%`; });
  $('shockApply').addEventListener('click', () => { applyShock().catch((e) => { $('shockOut').textContent = String(e.message || e); }); });
  $('shockReset').addEventListener('click', () => { clearShock(); render(); });
}

function fail(e) {
  console.error(e);
  $('status').textContent = `error: ${e.message || e}`;
  if (S.selftest) document.title = `fluxscape-error:${e.message || e}`;
}

(async function main() {
  try {
    if (typeof deck === 'undefined') throw new Error('deck.gl failed to load');
    wire();
    setArcsLabel();
    await onStatus(await getJSON('/api/status'));
    // Headless self-test: an open SSE stream is a pending request that stalls Chrome's virtual clock
    // (--virtual-time-budget never expires), so the self-test renders from the initial status only.
    if (S.selftest) return;
    const es = new EventSource('/api/events');
    es.addEventListener('status', (ev) => onStatus(JSON.parse(ev.data)).catch(fail));
    es.onerror = () => { $('status').textContent = 'reconnecting…'; };
  } catch (e) { fail(e); }
}());
