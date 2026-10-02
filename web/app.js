/* global deck */
'use strict';
const Q = new URLSearchParams(location.search);
const S = {
  status: null, times: [], frame: null, raster: null, shock: null, shockRaster: null,
  heightBase: 1, deck: null, playing: null, loadedKey: '', selftest: Q.has('selftest'),
  idx: null, idxKey: '',
};
const $ = (id) => document.getElementById(id);

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

function colormap(v, vmax) {
  const x = Math.max(-1, Math.min(1, vmax > 0 ? v / vmax : 0));
  const a = Math.abs(x), mid = [247, 247, 247], end = x < 0 ? [33, 102, 172] : [178, 24, 43];
  return [0, 1, 2].map((k) => Math.round(mid[k] + (end[k] - mid[k]) * a));
}

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

// Triangulated surface: one vertex per raster sample, Gouraud-coloured by z with the diverging colormap
// (per-vertex COLOR_0, float RGB 0..1, consumed by SimpleMeshLayer as the `colors` attribute; no texture).
function buildTerrain(z, meta, lattice, scale) {
  const { w, h } = meta, sx = lattice.cols / w, sy = lattice.rows / h;
  const pos = new Float32Array(w * h * 3), nrm = new Float32Array(w * h * 3), col = new Float32Array(w * h * 3);
  const vmax = Math.max(Math.abs(meta.zmin), Math.abs(meta.zmax), 1e-9);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = y * w + x;
    pos[3 * k] = (x + 0.5) * sx; pos[3 * k + 1] = (y + 0.5) * sy; pos[3 * k + 2] = z[k] * scale;
    const c = colormap(z[k], vmax);
    col[3 * k] = c[0] / 255; col[3 * k + 1] = c[1] / 255; col[3 * k + 2] = c[2] / 255;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = y * w + x;
    const dx = (z[y * w + Math.min(w - 1, x + 1)] - z[y * w + Math.max(0, x - 1)]) * scale / (2 * sx);
    const dy = (z[Math.min(h - 1, y + 1) * w + x] - z[Math.max(0, y - 1) * w + x]) * scale / (2 * sy);
    const len = Math.hypot(dx, dy, 1);
    nrm[3 * k] = -dx / len; nrm[3 * k + 1] = -dy / len; nrm[3 * k + 2] = 1 / len;
  }
  return {
    topology: 'triangle-list',
    attributes: { POSITION: { value: pos, size: 3 }, NORMAL: { value: nrm, size: 3 }, COLOR_0: { value: col, size: 3 } },
    indices: { value: gridIndices(w, h), size: 1 },
  };
}

function sample(z, meta, lattice, col, row) {
  const px = Math.min(meta.w - 1, Math.floor((col + 0.5) * meta.w / lattice.cols));
  const py = Math.min(meta.h - 1, Math.floor((row + 0.5) * meta.h / lattice.rows));
  return z[py * meta.w + px];
}

function render() {
  const f = S.frame;
  if (!f || !S.deck) return;
  const L = f.lattice;
  const meta = S.shock ? S.shock.raster : f.raster, z = S.shock ? S.shockRaster : S.raster;
  const scale = S.heightBase * parseFloat($('hscale').value);
  const nodes = f.nodes.map((n) => ({ i: n[0], ticker: n[1], sector: n[2], col: n[3] % L.cols, row: Math.floor(n[3] / L.cols), h: n[6], pi: n[8], score: n[9] }));
  const byI = new Map(nodes.map((n) => [n.i, n]));
  const pos = (n, lift = 0.3) => [n.col + 0.5, n.row + 0.5, sample(z, meta, L, n.col, n.row) * scale + lift];
  const ranked = nodes.filter((n) => n.h !== null).sort((a, b) => b.h - a.h);
  const labeled = ranked.slice(0, 30).concat(ranked.slice(-30));
  const arcs = $('arcs').checked ? f.arcs.slice(0, 400).map((a) => ({ s: byI.get(a[0]), t: byI.get(a[1]), w: a[2] })).filter((a) => a.s && a.t) : [];
  const wmax = arcs.reduce((m, a) => Math.max(m, a.w), 1e-12);
  const holdings = (f.portfolio || []).filter((p) => p.i !== null && byI.has(p.i)).map((p) => ({ ...byI.get(p.i), weight: p.weight }));
  const layers = [
    new deck.SimpleMeshLayer({
      id: 'terrain', data: [{}], mesh: buildTerrain(z, meta, L, scale),
      getPosition: [0, 0, 0], getColor: [255, 255, 255], material: { ambient: 0.55, diffuse: 0.55, shininess: 12, specularColor: [30, 30, 30] },
    }),
    new deck.ScatterplotLayer({
      id: 'nodes', data: nodes, getPosition: (n) => pos(n), getRadius: 0.16, radiusUnits: 'common',
      getFillColor: (n) => (n.h >= 0 ? [178, 24, 43, 220] : [33, 102, 172, 220]), pickable: true,
    }),
    new deck.ArcLayer({
      id: 'arcs', data: arcs, getSourcePosition: (a) => pos(a.s), getTargetPosition: (a) => pos(a.t),
      getWidth: (a) => 0.5 + 3 * a.w / wmax, widthUnits: 'pixels', getSourceColor: [255, 140, 0, 150], getTargetColor: [255, 215, 0, 150],
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
  const span = Math.max(lattice.cols, lattice.rows);
  S.deck = new deck.Deck({
    parent: el,
    views: new deck.OrbitView({ orbitAxis: 'Z', fovy: 40 }),
    initialViewState: { target: [lattice.cols / 2, lattice.rows / 2, 0], rotationX: 45, rotationOrbit: -25, zoom: Math.log2(Math.min(el.clientWidth, el.clientHeight) / (1.6 * span)), minZoom: -6, maxZoom: 12 },
    controller: true,
    getTooltip: ({ object }) => (object && object.ticker
      ? `${object.ticker} · ${object.sector}\nh ${object.h === null ? 'n/a' : object.h.toFixed(3)}  π ${object.pi === null ? 'n/a' : object.pi.toExponential(2)}\nscore+1 ${object.score === null ? 'n/a' : object.score.toFixed(3)}`
      : null),
  });
}

async function loadFrame(t) {
  const q = t === undefined || t === null ? '' : `?t=${t}`;
  const [f, z] = await Promise.all([getJSON(`/api/frame${q}`), getFloat32(`/api/frame/grid${q}`)]);
  S.frame = f; S.raster = z;
  if (!S.deck) {
    const zmax = Math.max(Math.abs(f.raster.zmin), Math.abs(f.raster.zmax), 1e-9);
    S.heightBase = 0.2 * Math.max(f.lattice.cols, f.lattice.rows) / zmax;
    initDeck(f.lattice);
    $('tickers').innerHTML = f.nodes.map((n) => `<option value="${n[1]}">`).join('');
  }
  $('tlabel').textContent = `${f.time}  ·  ${f.nodes.length} active stocks  ·  ${f.params}`;
  $('holdings').innerHTML = (f.portfolio || []).map((p) => {
    const n = p.i === null ? null : f.nodes.find((x) => x[0] === p.i);
    const h = n && n[6] !== null ? n[6] : null;
    const cls = h === null ? '' : (h >= 0 ? 'pos' : 'neg');
    return `${p.ticker.padEnd(6)} ${(100 * p.weight).toFixed(1).padStart(5)}%  <span class="${cls}">${h === null ? 'n/a' : h.toFixed(3)}</span>`;
  }).join('\n');
  render();
}

async function refreshTimes() {
  S.times = await getJSON('/api/times');
  const s = $('scrub');
  s.max = Math.max(0, S.times.length - 1);
  if ($('follow').checked) s.value = s.max;
}

function showStatus() {
  const st = S.status;
  if (!st) return;
  $('status').textContent = st.error ? `error: ${st.error}` : `${st.label} · ${st.ready ? 'ready' : `computing ${st.computed}/${st.total}`} · ${st.nodes} stocks`;
}

async function applyShock() {
  const body = { shocks: [{ ticker: $('shockTicker').value.trim().toUpperCase(), size: Number($('shockSize').value) }] };
  const r = await getJSON('/api/shock', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  S.shockRaster = await getFloat32('/api/shock/grid'); S.shock = r;
  const fmt = (x) => `${x.ticker.padEnd(6)} ${x.dh >= 0 ? '+' : ''}${x.dh.toFixed(4)}`;
  $('shockOut').textContent = `Δh landscape · L1 Δπ ${r.l1_dpi.toExponential(2)}\n` +
    `shocked: ${r.shocked.map(fmt).join(', ')}\n\nreceivers\n${r.receivers.slice(0, 10).map(fmt).join('\n')}\n\nlosers\n${r.losers.slice(0, 10).map(fmt).join('\n')}`;
  render();
}

async function onStatus(st) {
  S.status = st;
  showStatus();
  const key = `${st.generation}:${st.ready}`;
  if (st.ready && key !== S.loadedKey) {
    S.loadedKey = key;
    S.shock = null;
    await refreshTimes();
    await loadFrame(S.times[Number($('scrub').value)]);
    if (S.selftest) {
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
}

function wire() {
  $('scrub').addEventListener('input', () => { $('follow').checked = false; S.shock = null; loadFrame(S.times[Number($('scrub').value)]).catch(fail); });
  $('play').addEventListener('click', () => {
    if (S.playing) { clearInterval(S.playing); S.playing = null; $('play').textContent = 'Play'; return; }
    $('play').textContent = 'Pause';
    S.playing = setInterval(() => {
      const s = $('scrub'); s.value = (Number(s.value) + 1) % (Number(s.max) + 1);
      loadFrame(S.times[Number(s.value)]).catch(fail);
    }, 700);
  });
  ['hscale', 'arcs', 'labels'].forEach((id) => $(id).addEventListener('input', render));
  $('apply').addEventListener('click', async () => {
    const body = { preset: $('preset').value, height: $('height').value, idw_power: Number($('idwPower').value), idw_radius: Number($('idwRadius').value), subdivision: Number($('subdiv').value) };
    if ($('href').value) body.h_ref = $('href').value;
    try { await getJSON('/api/params', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); S.deck && S.deck.finalize(); S.deck = null; } catch (e) { fail(e); }
  });
  $('shockSize').addEventListener('input', () => { $('shockSizeLabel').textContent = `${$('shockSize').value}%`; });
  $('shockApply').addEventListener('click', () => { applyShock().catch((e) => { $('shockOut').textContent = String(e.message || e); }); });
  $('shockReset').addEventListener('click', () => { S.shock = null; $('shockOut').textContent = ''; render(); });
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
    await onStatus(await getJSON('/api/status'));
    // Headless self-test: an open SSE stream is a pending request that stalls Chrome's virtual clock
    // (--virtual-time-budget never expires), so the self-test renders from the initial status only.
    if (S.selftest) return;
    const es = new EventSource('/api/events');
    es.addEventListener('status', (ev) => onStatus(JSON.parse(ev.data)).catch(fail));
  } catch (e) { fail(e); }
}());
