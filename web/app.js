/* global deck */
'use strict';
const Q = new URLSearchParams(location.search);
const S = {
  status: null, times: [], frame: null, raster: null, shock: null, shockRaster: null, shockVmax: 1,
  baseVmax: 1, deck: null, playing: false, loadedKey: '', loadedGen: null, selftest: Q.has('selftest'),
  frameSeq: 0, shockSeq: 0, topSeq: 0, topPrev: new Map(), highlight: null, idx: null, idxKey: '', colCache: null, arcsInit: false, selftestDone: false,
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

// Robust scale: P90 of |v| (finite values only), floored at 1e-9. Colours saturate beyond ±vmax, so a few
// outliers can no longer wash out the rest of the map.
function p90abs(values) {
  const a = values.filter(Number.isFinite).map(Math.abs).sort((x, y) => x - y);
  return Math.max(a.length ? a[Math.min(a.length - 1, Math.floor(0.9 * a.length))] : 0, 1e-9);
}

// Height of a raster value: z·scale, soft-clipped with tanh at ±clip so outliers become plateaus, not needles.
function heightOf(v, scale, clip) { return clip * Math.tanh((v * scale) / clip); }

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
// SimpleMeshLayer as the `colors` attribute; no texture). Heights are heightOf(z, scale, clip).
function buildTerrain(z, meta, lattice, scale, vmax, clip) {
  const { w, h } = meta, sx = lattice.cols / w, sy = lattice.rows / h;
  const pos = new Float32Array(w * h * 3), nrm = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = y * w + x;
    pos[3 * k] = (x + 0.5) * sx; pos[3 * k + 1] = (y + 0.5) * sy; pos[3 * k + 2] = heightOf(z[k], scale, clip);
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

// ---- Holding arrows: from the smoothed surface to the holding's exact (unsmoothed) height ----
// Everything about the look lives here; swap arrowLayers() for another glyph (cylinder shaft, etc.) freely.
const ARROW = {
  up: [0, 170, 80, 255], down: [210, 40, 40, 255], flat: [120, 120, 120, 230],
  shaftPx: 7,          // shaft width in pixels
  minGapFrac: 0.02,    // gaps below this fraction of the span get a neutral marker instead of an arrow
};
// Arrowheads: billboarded triangles of a fixed pixel size (IconLayer), so the direction reads at any zoom even
// when the gap is short. The triangle's tip is anchored on the exact-height point.
const HEAD_PX = 28;
function headAtlas() {
  if (headAtlas.url) return headAtlas.url;
  // A white halo under a dark outline keeps a red head readable on red terrain (and a green one on green).
  const tri = (dx, pts, c) => `<g transform="translate(${dx} 0)" stroke-linejoin="round"><polygon points="${pts}" fill="white" stroke="white" stroke-width="9"/>` +
    `<polygon points="${pts}" fill="rgb(${c.slice(0, 3).join(',')})" stroke="rgb(25,25,25)" stroke-width="3"/></g>`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="128" height="64">${tri(0, '32,7 57,57 7,57', ARROW.up)}${tri(64, '7,7 57,7 32,57', ARROW.down)}</svg>`;
  headAtlas.url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  return headAtlas.url;
}
const HEAD_MAPPING = { up: { x: 0, y: 0, width: 64, height: 64, anchorY: 7 }, down: { x: 64, y: 0, width: 64, height: 64, anchorY: 57 } };

// items: holdings with x, y, zt (surface height), ez (exact height), dir ('up' | 'down' | 'flat').
function arrowLayers(items, params) {
  const arrows = items.filter((n) => n.dir === 'up' || n.dir === 'down');
  return [
    new deck.LineLayer({
      id: 'holding-arrow-shafts', data: arrows, getSourcePosition: (n) => [n.x, n.y, n.zt], getTargetPosition: (n) => [n.x, n.y, n.ez],
      getColor: (n) => ARROW[n.dir], getWidth: ARROW.shaftPx, widthUnits: 'pixels', pickable: true, parameters: params,
    }),
    new deck.IconLayer({
      id: 'holding-arrowheads', data: arrows, iconAtlas: headAtlas(), iconMapping: HEAD_MAPPING, getIcon: (n) => n.dir,
      getPosition: (n) => [n.x, n.y, n.ez], getSize: HEAD_PX, sizeUnits: 'pixels', billboard: true, pickable: true, parameters: params,
    }),
    new deck.ScatterplotLayer({
      id: 'holding-flat-markers', data: items.filter((n) => n.dir === 'flat'), getPosition: (n) => [n.x, n.y, n.zt + 0.2],
      getRadius: 5, radiusUnits: 'pixels', getFillColor: ARROW.flat, stroked: true, getLineColor: [255, 255, 255, 230],
      getLineWidth: 1.5, lineWidthUnits: 'pixels', pickable: true, parameters: params,
    }),
  ];
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
  const vmax = shock ? S.shockVmax : S.baseVmax;
  const scale = 0.25 * span(L) / vmax * hs, clip = 0.4 * span(L);
  const nodes = f.nodes.map((n) => {
    const col = n[3] % L.cols, row = Math.floor(n[3] / L.cols), s = sample(z, meta, L, col, row);
    const zt = heightOf(s.v, scale, clip);
    return { i: n[0], ticker: n[1], sector: n[2], x: s.x, y: s.y, zt, v: s.v, h: n[6], hd: n[7], pi: n[8], score: n[9], group: n[10], mr: n[11], pulse: n[12] };
  });
  const byI = new Map(nodes.map((n) => [n.i, n]));
  const pos = (n, lift = 0.3) => [n.x, n.y, n.zt + lift];
  // Label candidates: the 30 highest and lowest by the displayed height value (decluttered later).
  const ranked = nodes.filter((n) => n.hd !== null).sort((a, b) => b.hd - a.hd);
  const nArcs = shock ? 0 : Number($('arcs').value);
  const arcs = f.arcs.slice(0, nArcs).map((a) => ({ s: byI.get(a[0]), t: byI.get(a[1]), w: a[2] })).filter((a) => a.s && a.t);
  const wmax = arcs.reduce((m, a) => Math.max(m, a.w), 1e-12);
  const holdings = (f.portfolio || []).filter((p) => p.i !== null && byI.has(p.i)).map((p) => ({ ...byI.get(p.i), weight: p.weight }));
  const layers = [
    new deck.SimpleMeshLayer({
      id: 'terrain', data: [{}], mesh: buildTerrain(z, meta, L, scale, vmax, clip),
      getPosition: [0, 0, 0], getColor: [255, 255, 255], material: { ambient: 0.55, diffuse: 0.55, shininess: 12, specularColor: [30, 30, 30] },
    }),
    new deck.ScatterplotLayer({
      id: 'nodes', data: nodes, getPosition: (n) => pos(n), getRadius: 0.16, radiusUnits: 'common',
      getFillColor: (n) => signColor(shock ? n.v : n.hd, 220), pickable: true,
    }),
    new deck.ArcLayer({
      id: 'arcs', data: arcs, getSourcePosition: (a) => pos(a.s), getTargetPosition: (a) => pos(a.t),
      getWidth: (a) => 0.5 + 1.5 * a.w / wmax, widthUnits: 'pixels', getSourceColor: [255, 140, 0, 80], getTargetColor: [255, 215, 0, 80],
    }),
  ];
  // Portfolio: an arrow from the smoothed surface to each holding's exact height (base view only; in the shock
  // view the server sends no per-stock Δh for every holding, so the rings stay on the Δh surface). The ring and
  // ticker sit at the arrow tip. Drawn last without depth testing, so they stay visible at any lattice size.
  const onTop = { depthCompare: 'always', depthWriteEnabled: false };
  const ring0 = Math.max(0.6, 0.012 * span(L));
  const minGap = ARROW.minGapFrac * span(L);
  const hold = holdings.map((n) => {
    if (shock || n.hd === null || !Number.isFinite(n.hd)) return { ...n, ez: n.zt, dir: 'none' };
    const ez = heightOf(n.hd, scale, clip), gap = ez - n.zt;
    return { ...n, exact: n.hd, surface: n.v, ez, dir: Math.abs(gap) < minGap ? 'flat' : gap > 0 ? 'up' : 'down' };
  });
  if (!shock) layers.push(...arrowLayers(hold, onTop));
  layers.push(
    new deck.ScatterplotLayer({
      id: 'portfolio', data: hold, getPosition: (n) => [n.x, n.y, n.ez + 0.1], getRadius: (n) => ring0 + 1.5 * n.weight, radiusUnits: 'common',
      stroked: true, filled: false, getLineColor: [0, 230, 90, 255], getLineWidth: 3, lineWidthUnits: 'pixels', pickable: true, parameters: onTop,
    }),
  );
  S.holdingArrows = hold.map((n) => ({ ticker: n.ticker, exact: n.exact, surface: n.surface, dir: n.dir }));
  const hi = S.highlight === null ? undefined : byI.get(S.highlight);
  if (hi) {
    layers.push(new deck.ScatterplotLayer({
      id: 'highlight', data: [hi], getPosition: (n) => pos(n, 0.5), getRadius: 1.2 * ring0 + 0.4, radiusUnits: 'common',
      stroked: true, filled: false, getLineColor: [255, 210, 0, 255], getLineWidth: 4, lineWidthUnits: 'pixels', parameters: onTop,
    }));
  }
  S.baseLayers = layers;
  S.labelCands = {
    hold: hold.map((n) => ({ ticker: n.ticker, p: [n.x, n.y, n.ez], dir: n.dir })),
    others: $('labels').checked
      ? ranked.slice(0, 30).concat(ranked.slice(-30).reverse()).map((n) => ({ ticker: n.ticker, p: pos(n, 0.9) })) : [],
    onTop,
  };
  S.deck.setProps({ layers: [...layers, ...labelLayers()] });
  scheduleLabels();
}

// ---- Label decluttering: holdings first, then the highest, then the lowest values; a label is skipped when its
// screen box overlaps one already placed. At most LABEL_CAP labels. Re-run (debounced) after camera moves.
const LABEL_CAP = 25;
function labelLayers() {
  const c = S.labelCands;
  if (!c || !S.deck) return [];
  let vp;
  try { vp = S.deck.getViewports()[0]; } catch (e) { vp = undefined; }  // no view manager before the first frame
  const placed = [], hold = [], others = [];
  const overlaps = (b) => placed.some((q) => b[0] < q[2] && q[0] < b[2] && b[1] < q[3] && q[1] < b[3]);
  if (vp) {
    for (const l of c.hold) {  // holdings are always labelled; their boxes still block the others
      const [x, y] = vp.project(l.p);
      placed.push([x + 14, y - 9, x + 14 + 8.4 * l.ticker.length, y + 9]);
      const r = HEAD_PX / 2;  // the arrowhead hangs above (down arrow) or below (up arrow) the tip
      if (l.dir === 'down') placed.push([x - r, y - HEAD_PX, x + r, y]);
      if (l.dir === 'up') placed.push([x - r, y, x + r, y + HEAD_PX]);
      hold.push(l);
    }
    for (const l of c.others) {
      if (placed.length >= LABEL_CAP) break;
      const [x, y] = vp.project(l.p), w = 7.2 * l.ticker.length;
      const b = [x - w / 2 - 2, y - 15, x + w / 2 + 2, y + 1];
      if (overlaps(b)) continue;
      placed.push(b);
      others.push(l);
    }
  } else {
    hold.push(...c.hold);  // no viewport yet: holdings only; scheduleLabels() retries
  }
  return [
    new deck.TextLayer({
      id: 'labels', data: others, getPosition: (l) => l.p, getText: (l) => l.ticker, getSize: 12,
      getColor: [25, 25, 25], getTextAnchor: 'middle', getAlignmentBaseline: 'bottom', billboard: true,
    }),
    new deck.TextLayer({
      id: 'portfolio-labels', data: hold, getPosition: (l) => l.p, getText: (l) => l.ticker, getSize: 14,
      getColor: [0, 110, 45], fontWeight: 'bold', getTextAnchor: 'start', getAlignmentBaseline: 'center', getPixelOffset: [14, 0],
      billboard: true, outlineWidth: 3, outlineColor: [255, 255, 255, 230], fontSettings: { sdf: true }, parameters: c.onTop,
    }),
  ];
}
function scheduleLabels() {
  clearTimeout(S.labelTimer);
  S.labelTimer = setTimeout(() => {
    if (!S.deck || !S.baseLayers) return;
    S.deck.setProps({ layers: [...S.baseLayers, ...labelLayers()] });
    let ready = false;
    try { ready = S.deck.getViewports().length > 0; } catch (e) { ready = false; }
    if (!ready) scheduleLabels();  // retry until the first frame has created the viewport
  }, 120);
}

function initDeck(lattice) {
  const el = $('canvas');
  S.deck = new deck.Deck({
    parent: el,
    views: new deck.OrbitView({ orbitAxis: 'Z', fovy: 40 }),
    initialViewState: { target: [lattice.cols / 2, lattice.rows / 2, 0], rotationX: 45, rotationOrbit: -25, zoom: Math.log2(Math.min(el.clientWidth, el.clientHeight) / (1.6 * span(lattice))), minZoom: -6, maxZoom: 12 },
    controller: true,
    onViewStateChange: () => { scheduleLabels(); },
    getTooltip: ({ object }) => (object && object.ticker
      ? `${object.ticker} · ${object.sector}\n${clusterLine(object)}MarketRank π·N ${fmt(object.mr, 3)}  heartbeat ${fmtSigned(object.pulse, 4)}\nπ ${object.pi === null ? 'n/a' : object.pi.toExponential(3)}  h ${fmt(object.h, 3)}\ndisplayed height (${heightName().short}) ${fmtSigned(object.hd, 3)}${holdingLine(object)}`
      : null),
  });
}

// Holding tooltip line: exact (unsmoothed) value vs the smoothed surface under it.
function holdingLine(o) {
  if (o.exact === undefined || o.surface === undefined) return '';
  return `\nholding: exact ${fmtSigned(o.exact, 3)} · surface ${fmtSigned(o.surface, 3)} · Δ ${fmtSigned(o.exact - o.surface, 3)}`;
}

// "cluster #k · n stocks" (flux territories) or the sector's group size; loose stocks are the pooled remainder.
function clusterLine(o) {
  const f = S.frame;
  if (!f) return '';
  if (!S.groupSizes || S.groupSizes.frame !== f) {
    const m = new Map();
    f.nodes.forEach((n) => m.set(n[10], (m.get(n[10]) || 0) + 1));
    S.groupSizes = { frame: f, m };
  }
  const k = S.groupSizes.m.get(o.group) || 0;
  const flux = !S.status || S.status.territory !== 'sector';
  return o.group < 0 ? `loose · ${k} stocks\n` : `${flux ? 'cluster' : 'sector group'} #${o.group} · ${k} stocks\n`;
}

// What the landscape height is (the server's fixed landscape value), in one phrase.
const HEIGHTS = {
  pi_rel_size: { short: 'log π / size', text: 'Height: π relative to size, log(π / size share)', pos: 'more money than its size predicts', neg: 'less' },
  pi: { short: 'log π·N', text: 'Height: MarketRank, log(π·N)', pos: 'above average', neg: 'below' },
  hotness: { short: 'hotness', text: 'Height: hotness h (signed log)', pos: 'money accumulating', neg: 'draining' },
};
function heightName() { return HEIGHTS[(S.status && S.status.value) || 'pi_rel_size'] || HEIGHTS.pi_rel_size; }

function showLegend() {
  const box = $('legend');
  if (!S.frame) { box.replaceChildren(); return; }
  if (S.shock) {
    box.textContent = `Height: Δh after the shock, colour ±${S.shockVmax.toPrecision(3)} (P90)`;
    return;
  }
  const H = heightName();
  const sw = (c) => { const e = document.createElement('span'); e.className = 'sw'; e.style.background = `rgb(${c.join(',')})`; return e; };
  box.replaceChildren(document.createTextNode(H.text), sw(POS), document.createTextNode(H.pos), sw(NEG),
    document.createTextNode(`${H.neg} · colour ±${S.baseVmax.toPrecision(3)} (P90)`));
}

function fmt(v, d) { return v === null || v === undefined || !Number.isFinite(v) ? 'n/a' : v.toFixed(d); }
function fmtSigned(v, d) { const t = fmt(v, d); return t !== 'n/a' && v >= 0 ? `+${t}` : t; }

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
    // MarketRank π·N, coloured around 1 = average.
    const v = n ? n[11] : null;
    if (k) box.appendChild(document.createTextNode('\n'));
    box.appendChild(document.createTextNode(`${p.ticker.padEnd(6)} ${(100 * p.weight).toFixed(1).padStart(5)}%  π·N `));
    const el = document.createElement('span');
    if (v !== null && v !== undefined) el.className = v >= 1 ? 'pos' : 'neg';
    el.textContent = fmt(v, 3);
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

// ---- Heartbeat table: top 10 by exact MarketRank π ----
const SVGNS = 'http://www.w3.org/2000/svg';
function sparkline(series) {
  const W = 72, H = 18, pad = 2;
  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const pts = series.map((v, k) => [k, v]).filter((p) => p[1] !== null && Number.isFinite(p[1]));
  if (!pts.length) return svg;
  let lo = Infinity, hi = -Infinity;
  pts.forEach((p) => { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); });
  const n = Math.max(1, series.length - 1), range = hi - lo || 1;
  const xy = (p) => [pad + (W - 2 * pad) * p[0] / n, H - pad - (H - 2 * pad) * (hi === lo ? 0.5 : (p[1] - lo) / range)];
  const line = document.createElementNS(SVGNS, 'polyline');
  line.setAttribute('points', pts.map((p) => xy(p).map((c) => c.toFixed(1)).join(',')).join(' '));
  line.setAttribute('fill', 'none'); line.setAttribute('stroke', '#b2182b'); line.setAttribute('stroke-width', '1.2');
  svg.appendChild(line);
  const [cx, cy] = xy(pts[pts.length - 1]);
  const dot = document.createElementNS(SVGNS, 'circle');
  dot.setAttribute('cx', cx.toFixed(1)); dot.setAttribute('cy', cy.toFixed(1)); dot.setAttribute('r', '2.2'); dot.setAttribute('fill', '#b2182b');
  svg.appendChild(dot);
  return svg;
}

// Short sector names for the narrow table column; the full name goes in the cell's title.
const SECTOR_ABBR = {
  'Information Technology': 'InfoTech', 'Consumer Discretionary': 'ConsDisc', 'Communication Services': 'CommSvc',
  'Health Care': 'HealthCare', Financials: 'Financials', Industrials: 'Industrials', Energy: 'Energy', Materials: 'Materials',
  'Real Estate': 'RealEst', Utilities: 'Utilities', 'Consumer Staples': 'ConsStap', 'ETF/Fund': 'ETF', Unclassified: 'Unclass.',
};
function sectorAbbr(s) { return SECTOR_ABBR[s] || s; }

function cell(cls, text) { const d = document.createElement('span'); d.className = cls; d.textContent = text; return d; }

// The table's metric: π·N (`mr`). The sparkline (series of π·N) and the pulse animation follow it.
function topValue(r) { return r.mr; }

function renderTop(rows) {
  const box = $('topRows');
  const els = rows.map((r) => {
    const row = document.createElement('div');
    row.className = 'toprow' + (S.highlight === r.i ? ' sel' : '');
    const mv = r.prev_rank === null ? ['new', 'new'] : r.prev_rank > r.rank ? ['▲', 'up'] : r.prev_rank < r.rank ? ['▼', 'down'] : ['•', 'same'];
    row.title = `${r.prev_rank === null ? 'new in the ranking' : `previous rank ${r.prev_rank}`} · MarketRank π·N ${fmt(r.mr, 4)} · heartbeat Δlog π ${fmtSigned(r.pulse, 5)} · h ${fmt(r.h, 4)}`;
    const v = topValue(r);
    const pl = r.pulse === null || r.pulse === undefined ? 'same' : r.pulse > 0 ? 'up' : r.pulse < 0 ? 'down' : 'same';
    row.append(cell('rk', String(r.rank)), cell(`mv ${mv[1]}`, mv[0]), cell('tk', r.ticker), Object.assign(cell('sec', sectorAbbr(r.sector)), { title: r.sector }),
      cell('hv', fmt(v, 3)), cell(`pl ${pl}`, fmtSigned(r.pulse, 4)), sparkline(r.series));
    row.addEventListener('click', () => {
      S.highlight = S.highlight === r.i ? null : r.i;
      box.querySelectorAll('.toprow').forEach((el) => el.classList.remove('sel'));
      if (S.highlight === r.i) row.classList.add('sel');
      render();
    });
    if (S.topPrev.has(r.i) && S.topPrev.get(r.i) !== v) row.classList.add('beat');
    return row;
  });
  box.replaceChildren(...els);
  S.topPrev = new Map(rows.map((r) => [r.i, topValue(r)]));
}

// Refreshes the table for frame t; stale responses (superseded by a newer frame) are dropped.
async function loadTop(t) {
  const seq = ++S.topSeq;
  const r = await getJSON(`/api/top?n=10&bars=30&by=pi&t=${t}`);
  if (seq !== S.topSeq) return;
  renderTop(r.rows);
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
  // Base colour/height scale: P90 of |z| over the occupied vertices (stock cells). Under value = pi the stocks at
  // the teleport floor (all tied) are left out, so the floor plain does not set the scale.
  const floorTie = S.status && S.status.value === 'pi';
  S.baseVmax = p90abs(f.nodes.filter((n) => !(floorTie && n[13])).map((n) => sample(z, f.raster, f.lattice, n[3] % f.lattice.cols, Math.floor(n[3] / f.lattice.cols)).v));
  showStatus();
  if (!S.deck) {
    initDeck(f.lattice);
    fillTickers(f.nodes);
  }
  if (!S.arcsInit) { S.arcsInit = true; $('arcs').value = Math.min(2 * f.nodes.length, 150); setArcsLabel(); }
  fillHoldings(f);
  loadTop(f.t).catch((e) => { $('topRows').textContent = String(e.message || e); if (S.selftest) fail(e); });
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

// "MarketRank · replay 1d · N stocks · <date>": no parameter dump.
function showStatus() {
  const st = S.status;
  if (!st) return;
  const parts = ['MarketRank', st.label];
  if (st.error) parts.push(`error: ${st.error}`);
  else if (!st.ready || !S.frame) parts.push(`computing ${st.computed}/${st.total}`);
  else {
    const funds = S.frame.nodes.reduce((k, n) => k + (n[2] === 'ETF/Fund' ? 1 : 0), 0);
    parts.push(`${S.frame.nodes.length} active (${S.frame.nodes.length - funds} stocks, ${funds} ETF/funds)`, S.frame.time.slice(0, 10));
  }
  if (S.shock) parts.push(`shock ${S.shock.shocked.map((x) => x.ticker).join(',')}`);
  $('status').textContent = parts.join(' · ');
  showLegend();
}

async function applyShock() {
  if (!isLatestFrame()) throw new Error('shock applies to the latest bar');
  const seq = ++S.shockSeq;
  const tickers = [$('shockTicker').value.trim().toUpperCase()];
  const body = { shocks: tickers.map((ticker) => ({ ticker, size: Number($('shockSize').value) })) };
  const r = await getJSON('/api/shock', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const z = await getFloat32(`/api/shock/grid?id=${r.shock_id}`);
  if (seq !== S.shockSeq) return;
  // Height and colour scale: P90 of |Δh| over active, non-shocked stocks (the shocked stock saturates).
  // (The receiver and loser lists from the server already exclude the shocked stocks.)
  const L = S.frame.lattice, shocked = new Set(tickers);
  S.shockVmax = p90abs(S.frame.nodes.filter((n) => !shocked.has(n[1]))
    .map((n) => sample(z, r.raster, L, n[3] % L.cols, Math.floor(n[3] / L.cols)).v));
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
    setTimeout(() => {
      if (!$('topRows').querySelector('.toprow')) fail(new Error('top table did not render'));
      // The product UI is fixed: no model or landscape knobs may come back.
      else if (document.querySelector('#panel select, #apply, #top button')) fail(new Error('a removed control is present'));
      else document.title = `marketrank-ok:${S.frame.nodes.length}`;
    }, 1500);
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
  $('shockSize').addEventListener('input', () => { $('shockSizeLabel').textContent = `${$('shockSize').value}%`; });
  $('shockApply').addEventListener('click', () => { applyShock().catch((e) => { $('shockOut').textContent = String(e.message || e); }); });
  $('shockReset').addEventListener('click', () => { clearShock(); render(); });
}

function fail(e) {
  console.error(e);
  $('status').textContent = `error: ${e.message || e}`;
  if (S.selftest) document.title = `marketrank-error:${e.message || e}`;
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
