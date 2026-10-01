# Fluxscape — Market Flux Graph, Steady-State Landscape & Shadow-Portfolio Optimizer

**Date:** 2026-10-01
**Status:** Draft — awaiting review
**Author:** kkaramete (with Claude)

## 1. Purpose

Fluxscape is a C++ program that models the market as a directed, weighted graph in which edge weights are the *estimated money flux* leaving one stock (net selling) and entering another (net buying). It solves the Markov chain defined by that graph for its steady state — exactly as PageRank does — and treats the resulting stationary probability as each stock's "heat". No predictive model is fitted: the market's own buy/sell heartbeat defines the chain.

The steady state is rendered as a 3D topological landscape (hills = money accumulating, valleys = money draining) in a deck.gl canvas. A user portfolio is placed on that landscape and an optimizer proposes a *collective* rebalancing move — uphill — at hourly, daily and weekly horizons. Every proposal is executed in a **shadow (paper) ledger** at realistic next-bar prices, so the program continuously measures whether its own advice would have made money, and keeps the full history across every parameter change.

**Non-goals:** placing real orders; fitted forecasting models (VAR, ML); intraday tick-level data; guarantees of profit. Fluxscape is advisory and experimental. The flux is an *inference* from price/volume co-movement — trades are anonymous, so true stock-to-stock order flow is not observable.

## 2. Decisions made during brainstorming

| Topic | Decision |
|---|---|
| Market data | Alpaca Market Data API v2 (REST bars) |
| Solver | Native C++ only (no Kinetica dependency); same math as Kinetica's PROBABILITY_RANK / PAGE_RANK |
| Output | Advisory only — no order placement |
| Portfolio input | Edited in UI, persisted to `data/portfolio.json` |
| Node placement | Force-directed layout on the flux graph, snapped to an n×m lattice |
| Forecast | Chain propagation π·P^k plus steady-state drift; no fitted parameters |
| Architecture | Single C++20 binary serving REST + SSE and the static web UI |
| Push channel | Server-Sent Events (one-way); controls via REST |
| Tracking | Append-only shadow ledger, one book per (strategy version × horizon) |

## 3. Universe

- `data/universe/sp500.csv` — `ticker,name,sector` for S&P 500 constituents. Bundled list is compiled from memory and may be stale; it is a plain editable CSV.
- **Portfolio extras:** any held ticker that is a single stock not in the CSV (e.g. **NVO**, Novo Nordisk ADR) is added as an extra graph node with sector `"Extra"`.
- **Index funds (look-through):** `data/universe/funds.csv` declares fund tickers and the universe they track (`VOO,sp500`). A fund is **not** a graph node. Its hotness is the weight-averaged hotness of its constituents, using trailing 20-bar average dollar volume as the weight proxy (Alpaca does not provide market cap). Fund prices are still fetched for valuation and P&L.

N ≈ 500–505 graph nodes.

## 4. Market data

### 4.1 Alpaca client
- `GET https://data.alpaca.markets/v2/stocks/bars` with `symbols` (batched, ≤ 100 per request), `timeframe`, `start`, `end`, `limit=10000`, `adjustment=all`, `feed`, `page_token` paging.
- Auth headers `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY` read from `.env` (gitignored) or environment.
- Default `feed=sip` with `end ≤ now − 16 min` (the free plan allows consolidated SIP history only beyond the 15-minute delay). `feed=iex` is configurable but discouraged: IEX volume is a small share of the market and distorts flux.
- HTTP 429 / 5xx: exponential backoff (0.5 s → 30 s, jittered), max 6 retries; then the affected symbols are marked **stale** for that frame.

### 4.2 Timeframes
| UI timeframe | Source | Notes |
|---|---|---|
| 1h | `30Min` bars aggregated into session-aligned hours 09:30–10:30 … 15:30–16:00 ET | Avoids Alpaca's clock-aligned 1Hour bars mixing pre-market into the 09:00 bar; extended hours excluded |
| 1d | `1Day` | |
| 1w | `1Week` | |

US Eastern time handling uses an explicit DST rule (2nd Sunday of March → 1st Sunday of November), not a tz database. NYSE holidays come from `data/universe/holidays.csv`.

### 4.3 Bar store & cache
- In-memory per-ticker bar series, plus on-disk cache `data/cache/<timeframe>/<ticker>.csv`. Fetches only request the missing tail.
- Restarts and parameter tweaks never re-download history.

### 4.4 Offline modes
- **Replay:** run from cached bars only.
- **Synthetic:** a built-in generator (seeded) produces a market of sectors with planted rotations; used when no keys/cache exist and for tests. The UI shows a banner stating the active data mode.

## 5. Flux graph model

For each bar *t* and stock *i* with close C, previous close C⁻ and volume V:

- **Return** `r_i = C/C⁻ − 1`; **dollar volume** `D_i = V · VWAP` (fallback C).
- **Pressure** `p_i = r_i · D_i`. `p_i < 0` → net selling (source); `p_i > 0` → net buying (sink).
- **Per-bar flux** for source *i*, sink *j*:
  `f_ij(t) = |p_i| · (p_j · a_ij) / Σ_{k∈sinks} (p_k · a_ik)`
  with **affinity** `a_ij = 1 + λ · max(0, ρ_ij)`, ρ the rolling Pearson correlation of returns over the last W bars. Each source distributes exactly its own outflow, so Σ_j f_ij = |p_i| (conservation). λ = 0 gives the pure heartbeat model.
- **Accumulation:** dense N×N matrix `F ← δ·F + f(t)`, δ = 2^(−1/halflife). Dense storage is cheap at N = 500 (≈2 MB).
- **Pruning:** keep the top-k outgoing edges per row (default k = 20) → CSR sparse matrix. k = N keeps the full matrix.
- **Transition matrix:** `P = rownorm(F_pruned)`. A row with zero outflow (pure accumulator) gets a self-loop of weight 1.
- **Damping / teleport:** `P' = α·P + (1−α)·(1/N)·𝟙` (default α = 0.85), guaranteeing a unique stationary distribution.
- **Steady state:** power iteration `π ← π·P'`, warm-started from the previous frame, until ‖Δπ‖₁ < 1e-10 or 1000 iterations (residual reported if not converged).
- **Hotness:** `h_i = N·π_i − 1` (0 = neutral, > 0 hill, < 0 valley).
- Stocks with missing/stale data in the current bar contribute `p_i = 0`; stocks with no data for the whole window are excluded from the frame.

### 5.1 Forecast (no fitted model)
π_t is by construction stationary for P'_t, so propagating it through the same matrix is a no-op (π·P'^k = π). The forecast therefore contrasts two time scales of the same flux:
- **Slow accumulator** F_slow (half-life 20 bars) → P'_slow → π_t, the equilibrium "landscape".
- **Fast accumulator** F_fast (half-life 3 bars) → P'_fast, the most recent flux pattern.
- Propagation: `π^(k) = π_t · P'_fast^k` for k ∈ {1, 4, 8} bars — "where the newest flux carries the equilibrium mass next".
- Drift: `d_i = π_t,i − π_{t−1},i`.
- **Forecast score** used by the optimizer: `s_i = N·(π^(k)_i − π_t,i) + β·N·d_i` (β default 0.5), where k is the horizon's step count.

## 6. Geometry — layout, lattice, landscape

1. **Force-directed layout** (Fruchterman–Reingold, O(N²) per iteration — fine at N ≈ 500) on the symmetrized pruned flux graph (attraction ∝ F_ij + F_ji, global repulsion). Warm-started from the previous frame's positions; 50–200 iterations with cooling.
2. **Lattice snap:** lattice n×m with n·m ≥ N, near-square (500 → 23×22). Stocks are assigned to lattice points by the **Hungarian algorithm**, minimizing total squared distance from normalized layout positions. **Hysteresis:** the new assignment is adopted only if its cost improves on the previous assignment (re-evaluated against new positions) by more than τ (default 5%); otherwise cells are kept. Empty lattice points take IDW values from their neighbors.
3. **IDW landscape:** values h at lattice points interpolated onto a fine raster (subdivision s, default 4 → ~90×90) with `z(x) = Σ w_i h_i / Σ w_i`, `w_i = 1/d_i^q` (power q default 2), restricted to a search radius R (default 3 lattice cells; nearest-only fallback if no point in radius). Exact at nodes; bounded by [min h, max h].
4. **Raw-layout toggle:** IDW over the un-snapped layout positions, for comparison.

Graph adjacency ≠ geometric adjacency: the layout approximates flux proximity in 2D, so strong edges may still span the map. Flux edges are therefore drawable as arcs over the terrain.

## 7. Portfolio optimizer (collective move)

### 7.1 Portfolio on the landscape
A portfolio is a set of weighted points on the terrain. Its **altitude** is `A(w) = Σ_i w_i · h_i` (funds use look-through hotness). The optimizer moves the portfolio mass uphill *collectively*, trading off forecast gain against concentration risk and transaction cost.

### 7.2 Problem (per horizon H ∈ {hourly, daily, weekly})
Candidate set C = current holdings ∪ top-M nodes by forecast score s (M default 25) ∪ cash. Variables: target weights w ∈ ℝ^|C|.

```
maximize    sᵀw  −  η · wᵀΣw  −  γ · ‖w − w₀‖₁
subject to  Σ w = 1,   0 ≤ w_i ≤ cap_i,   ‖w − w₀‖₁ ≤ 2τ_H
```
- `s`: forecast scores for horizon H (§5.1); cash has s = 0, zero variance.
- `Σ`: rolling covariance of returns over W bars of H's timeframe (shrunk 50% toward diagonal for stability).
- `w₀`: current shadow-book weights; γ: cost penalty (linked to slippage bps); τ_H: max one-way turnover per rebalance (defaults hourly 5%, daily 15%, weekly 30%).
- `cap_i`: per-name cap on *increases*: `cap_i = max(cap, w₀,i)` with cap default 35% for single stocks and 100% for index funds and cash. A position already above the cap (e.g. AAPL at 60%) is never force-sold because of the cap; it can only be reduced when the score justifies it, and it cannot grow.
- Short selling and leverage are not allowed.

It is a small convex problem (≤ ~40 variables) solved in C++ by **ADMM** (QP block + capped-simplex projection + L1 prox), with a convergence check. A brute-force grid solver validates it on tiny instances in tests.

**Output — a proposal:** target weights, the resulting buy/sell trade list (shares and $), expected altitude change ΔA, turnover, and a per-trade rationale (e.g. "AAPL h=−0.4 draining into NVDA/AVGO cluster"). Trades below $500 are suppressed.

### 7.3 Schedule
- Hourly: at each session-aligned hourly bar close.
- Daily: at the daily bar close.
- Weekly: at the weekly bar close (last trading day of the week).

## 8. Shadow ledger & performance tracking

### 8.1 Initial portfolio (made-up, editable)
$1,000,000 at inception, allocated by weight and converted to fractional shares at the inception bar close:

| Ticker | Weight | Notes |
|---|---|---|
| AAPL | 60% | heavy position |
| VOO | 15% | Vanguard S&P 500 ETF (look-through) |
| NVDA | 7% | |
| LLY | 6% | |
| NVO | 5% | not in S&P 500 → extra node |
| NKE | 3.5% | |
| F | 3.5% | |

### 8.2 Books
- **Baseline:** buy-and-hold of the initial portfolio — never trades.
- **Benchmark:** 100% VOO.
- **Strategy books:** one per (strategy version × horizon). Each starts at the same $1M initial portfolio and follows only its own horizon's proposals.

### 8.3 Fill model (no look-ahead)
A proposal is computed from data up to the close of bar t and filled at the **open of bar t+1** for that timeframe, with slippage (default 5 bps per side) and zero commission. Fractional shares are allowed. Funds and stocks use the same split/dividend-adjusted prices, so returns approximate total return.

### 8.4 Strategy versions — history survives every tweak
- A **strategy version** is an immutable snapshot of all parameters affecting proposals (λ, half-life, k, α, β, η, γ, τ, M, caps, slippage). Its id is a content hash plus a user-supplied label.
- Changing any of those parameters in the UI does not edit the running strategy. It creates a **new version**: a *challenger*. The previous versions keep running forward, so champion and challengers are tracked side by side.
- Each new version is immediately **backtested** over the cached lookback (walk-forward, same fill model) so it can be compared before it has live history. Backtest results are stored separately from forward results and labelled as such.
- Display-only parameters (IDW power, colormap, subdivision, etc.) do not create versions.

### 8.5 Storage
Append-only JSON Lines under `data/ledger/` (human-readable, crash-safe by append; not gitignored):
- `strategies.jsonl` — version id, label, params, created_at, parent version.
- `proposals.jsonl` — book, bar time, candidate scores, target weights, trades, ΔA.
- `fills.jsonl` — book, fill time, ticker, side, shares, price, slippage.
- `nav.jsonl` — book, bar time, cash, positions, NAV.

On startup the server **catches up**: it replays every bar missed while it was down for every active book, deterministically from cached data, so tracking has no gaps.

### 8.6 Metrics (per book, forward and backtest separately)
Cumulative return; excess return vs. baseline and vs. VOO; annualized volatility; Sharpe (rf = 0); max drawdown; hit rate (share of proposals whose filled trades beat holding w₀ over the next bar); turnover; number of trades. "Steady profit" is judged by excess return *and* drawdown together, not return alone.

## 9. Runtime

### 9.1 Source layout
```
CMakeLists.txt            C++20; FetchContent: cpp-httplib (OpenSSL), nlohmann/json, doctest
src/
  market/   universe, alpaca_client, bar_store, session_calendar, synthetic_market
  graph/    flux_builder, pruner (CSR), markov_solver, forecaster
  geom/     force_layout, lattice_assigner (Hungarian + hysteresis), idw_grid
  optimize/ admm_portfolio, proposal
  ledger/   strategy_registry, shadow_book, ledger_store (jsonl), metrics
  server/   http_server (REST + SSE + static), pipeline (frame scheduler)
  main.cpp
web/        index.html, app.js, style.css (deck.gl UMD from CDN, no build step)
tests/      doctest unit + integration
data/       universe/, cache/ (ignored), ledger/, portfolio.json
```

### 9.2 Frame pipeline (one Δt)
`bars(t) → flux update → prune → solve π → forecast → layout → lattice snap → IDW → optimize (if horizon closes) → fill previous proposals → NAV → publish`

Changing a parameter re-runs only the affected stage and those after it. Replay speed is clamped so that the frame interval ≥ max(user Δt, 1.5 × measured compute time), so larger graphs automatically slow playback rather than queuing frames.

### 9.3 API
| Endpoint | Purpose |
|---|---|
| `GET /api/frame/latest` (or `?t=`) | JSON: nodes (ticker, sector, π, h, forecasts, cell, raw xy), pruned edges, portfolio markers |
| `GET /api/frame/latest/grid` | Binary Float32 raster + header (w, h, min, max) |
| `GET /api/events` | SSE: `frame`, `proposal`, `status`, `error` |
| `GET/POST /api/params` | all parameters; POST of strategy parameters creates a new version |
| `GET/PUT /api/portfolio` | initial portfolio definition |
| `GET /api/books`, `GET /api/books/{id}/nav` | books, metrics, NAV curves |
| `GET /api/proposals?book=` | proposal history with fills |
| `GET /api/strategies` | version tree |
| `POST /api/replay` | play / pause / seek / speed |

### 9.4 UI
**Left panel (collapsible sections):**
- *Data:* timeframe (1h/1d/1w), lookback, live/replay toggle, scrubber, play, speed, data-mode banner.
- *Graph:* λ, half-life, top-k, α, β; forecast view (now / +1 / +4 / +8).
- *Landscape:* lattice n×m (auto, can be overridden), subdivision, IDW power and radius, height exaggeration, colormap, flux arcs on/off and threshold, raw-layout toggle, labels.
- *Portfolio & Strategy:* holdings table (editable initial portfolio), current shadow-book positions per horizon, latest proposal with trade list and rationale, strategy version selector with "save as new version" and label, optimizer params (η, γ, τ, M, caps).
- *Performance:* NAV chart of baseline, VOO and strategy books (forward solid, backtest dashed); metrics table.

**Canvas (deck.gl `OrbitView`, Cartesian):** terrain `SimpleMeshLayer` built from the raster (computed normals, diverging blue→red colormap); stock markers (`ScatterplotLayer`) and labels (`TextLayer`); flux `ArcLayer`; **portfolio markers** — held stocks as rings sized by weight, VOO as a translucent plane at its look-through altitude, and proposed moves as arrows from sells to buys. Hover tooltip: ticker, sector, π, h, top-3 in/out flux, forecast score, held weight. New frames animate positions and heights.

## 10. Error handling

- No API keys → replay from cache, or synthetic market; banner shows which.
- Rate limit / HTTP errors → backoff; failing symbols marked stale and excluded from the frame (never zero-filled prices). A book holding a stale symbol carries its last price and the NAV is flagged.
- Solver non-convergence → iteration cap, residual shown in the status bar; the proposal for that frame is skipped (and logged) rather than produced from an unconverged π.
- ADMM non-convergence → no proposal; logged.
- Market closed / holiday → no new bars; live mode shows the last frame; no proposals.
- Ledger write failure → pipeline halts proposals and surfaces an error (tracking integrity beats availability).

## 11. Testing

Unit (doctest):
- Flux: per-source conservation; every row of P sums to 1; accumulator self-loop; λ = 0 equivalence with the pure model.
- Solver: analytic stationary distribution of 3- and 4-state chains; warm start reaches the same π; damping guarantees convergence on a reducible chain.
- Hungarian: equals brute force for n ≤ 7; hysteresis blocks sub-threshold swaps.
- IDW: exact at nodes; output within [min, max]; radius fallback.
- ADMM: KKT/feasibility (sum = 1, bounds, turnover); matches the brute-force grid on 3-asset cases; zero-signal input yields no trade (cost penalty).
- Ledger: fill at next-bar open, never same-bar; slippage applied; NAV identity (cash + Σ shares·price); catch-up replay is deterministic (same ledger twice).
- Session calendar: DST transitions; 30Min → session-hour aggregation; holidays.
- Alpaca: parsing and paging from recorded JSON fixtures.

Integration:
- A synthetic market with a planted rotation (sector A → sector B) makes sector B the top hill, places it in a contiguous lattice patch, and makes the daily strategy book rotate into B and beat the baseline in that synthetic run.
- End-to-end server smoke test: start in synthetic mode, fetch frame JSON, grid and books.

## 12. Implementation milestones

1. **Core model (CLI):** universe, synthetic market, Alpaca client and cache, flux → solver → forecast; CLI prints top hills and valleys.
2. **Geometry + server + UI:** layout, lattice, IDW, REST/SSE, deck.gl landscape and left-panel controls.
3. **Optimizer + ledger:** ADMM, shadow books, strategy versions, backtest, catch-up, performance panel.

## 13. Caveats

- The flux is inferred from simultaneous price and dollar-volume moves; it is not observed order flow.
- Hotness-to-return is not a calibrated relationship; the shadow ledger exists precisely to measure whether it works. Past shadow performance does not imply future results.
- The bundled S&P 500 list may be out of date.
- The free Alpaca plan means SIP data runs about 15 minutes behind the market.
