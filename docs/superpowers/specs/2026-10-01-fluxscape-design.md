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

Two universe sources, selected by `--universe sp500|snapshot` (default: the latest snapshot if one exists, otherwise sp500):

- **sp500:** `data/universe/sp500.csv` (`ticker,name,sector`, refreshed from Wikipedia by `scripts/fetch_sp500.py`).
- **snapshot (liquidity-ranked top N, default N = 10000):** built in alpaca mode with `--universe-size N`:
  1. Fetch all assets from Alpaca's trading API `GET /v2/assets?status=active&asset_class=us_equity`.
  2. Keep assets that are `tradable` and listed on NYSE, NASDAQ, ARCA, NYSEARCA, AMEX or BATS.
  3. Drop warrants, units and rights by name (case-insensitive `warrant`, word `unit(s)`, word `right(s)`). ETFs are kept as ordinary nodes (user decision 2026-10-01). S&P 500 constituents always pass, and `data/universe/exclude.csv` / `include.csv` (one ticker per line) override the rules.
  4. Rank by the median daily dollar volume (v·vwap) over the last 20 trading days and keep the top N.
  5. Write `data/universe/universe_<YYYY-MM-DD>.csv` (`ticker,name,sector,exchange,median_dollar_volume`); a snapshot younger than 7 days is reused. Sector comes from sp500.csv where known, otherwise `Unclassified`.
- **Portfolio extras:** any held single stock not in the universe (e.g. **NVO**) is added as an extra node with sector `"Extra"`.
- **Index funds (look-through):** `data/universe/funds.csv` declares fund tickers and what they track (`VOO,sp500`). A fund that is also in the ranked universe is an ordinary node too; look-through is used only where the fund isn't a node. Where a fund is not a node, its hotness is the weight-averaged hotness of its constituents, weighted by trailing 20-bar median dollar volume (Alpaca does not provide market cap). Fund prices are still fetched for valuation and P&L.

N ranges from ~505 (sp500) to ~10,000 (snapshot). Every per-bar data structure is O(N·k) or O(N·W); none is O(N²).

## 4. Market data

### 4.1 Alpaca client
- `GET https://data.alpaca.markets/v2/stocks/bars` with `symbols` (batched, ≤ 100 per request), `timeframe`, `start`, `end`, `limit=10000`, `adjustment=all`, `feed`, `page_token` paging.
- Auth headers `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY` read from `.env` (gitignored) or environment.
- Default `feed=sip` with `end ≤ now − 16 min` (the free plan allows consolidated SIP history only beyond the 15-minute delay). `feed=iex` is configurable but discouraged: IEX volume is a small share of the market and distorts flux.
- HTTP 429 / 5xx: exponential backoff (0.5 s → 30 s, jittered), max 6 retries; then the affected symbols are marked **stale** for that frame.
- Client-side rate limit: at most 180 requests/minute (a minimum interval of 334 ms between requests), under Alpaca's 200/min, so a 10,000-ticker sync is never throttled.
- Assets come from the trading API host (`paper-api.alpaca.markets` by default, overridable with `APCA_TRADING_HOST`), with the same credentials.

### 4.2 Timeframes
| UI timeframe | Source | Notes |
|---|---|---|
| 1h | `30Min` bars aggregated into session-aligned hours 09:30–10:30 … 15:30–16:00 ET | Avoids Alpaca's clock-aligned 1Hour bars mixing pre-market into the 09:00 bar; extended hours excluded |
| 1d | `1Day` | |
| 1w | `1Week` | |

US Eastern time handling uses an explicit DST rule (2nd Sunday of March → 1st Sunday of November), not a tz database. NYSE holidays come from `data/universe/holidays.csv`.

### 4.3 Bar store: DuckDB + Hive-partitioned Parquet lake
- **Engine:** embedded DuckDB (prebuilt `libduckdb` v1.5.6, called from C++ as plain SQL through `duckdb::Connection::Query` and `duckdb::Appender`). No extensions; Parquet support is built in.
- **Layout** (`data/lake/`):
  - `bars/tf=<1h|1d|1w>/year=<Y>/month=<M>/part-<uuid>.parquet`, with columns `ticker, t, o, h, l, c, v, vw, seq` (lossless DOUBLE values).
  - `catalog.duckdb` with small native tables: `meta` (seq counter), `coverage(ticker, tf, covered_from)`, `complete(ticker, tf, t)` (the latest bar stored after its session or bucket had closed; only moves later), and `pending` / `pending_cov` / `pending_complete` (write-ahead buffers).
  - `_staging/` for in-flight writes.
  - Partition values come from the UTC bar time. The layout is the data layout Iceberg uses, so a later catalog registration needs no rewrite. Full Iceberg was evaluated and deferred: in DuckDB 1.5.6, writes need a REST catalog server, and snapshot expiry needs a separate tool.
- **Writes:** each flush is one batch with a new `seq`. Rows go to `pending` through the Appender, then `COPY ... TO _staging/<id> (FORMAT parquet, PARTITION_BY (tf, year, month))`, then the files are renamed into `bars/` (atomic on the same filesystem). After that, `pending` is cleared and coverage is applied, so coverage never runs ahead of the data. On open, leftover staging is deleted and leftover `pending` rows are republished.
- **Upserts:** a re-fetched bar is written again with a higher seq. Every read keeps the newest version: `QUALIFY row_number() OVER (PARTITION BY ticker, t ORDER BY seq DESC) = 1`.
- **Reads:** a windowed `read_parquet(..., hive_partitioning=true)` filtered on year, t and the requested tickers, with partition pruning. The working set and the panel are bounded by the lookback window.
- **Single writer:** DuckDB's lock on `catalog.duckdb` stops a second process from opening the lake.
- **Adjustment check:** the tail fetch starts at `min(last stored bar, complete_through)`. Every re-fetched bar at or before `complete_through` is compared with its stored close. A bar counts as complete if its session (1d: the session calendar's 16:00 ET close) or its bucket (intraday: bar start + timeframe; 1w: start + 7 days) had closed by the fetch end of the sync that stored it, and the `complete` table records this.
  - **Re-adjusted history:** a difference above 1e-6 relative means the history was re-adjusted, for example by a split or dividend. The ticker's tail is then discarded, not merged, and the whole history is fetched again from `min(start, first stored bar, covered_from)` through end. The higher seq wins.
  - **Failed refetch:** the ticker stays untouched and stale, so the next sync detects the change again.
  - **Partial bars:** a bar stored before its close can change legitimately, so it is never compared. If no complete stored bar overlaps the fetch, the check is skipped.
  - **`--refetch-full`:** a one-time repair that runs the full refetch for every ticker in the sync.
- **Commit granularity:** `sync_bars` commits after every 100-symbol fetch batch, so an interrupted sync loses at most one batch.
- **Maintenance** (after every alpaca sync, and via `--maintain`):
  - **Compaction:** a partition with more than 8 files is rewritten into one deduplicated file.
  - **Retention:** `data/lake/retention.json` (default `{"1h": 730, "1d": null, "1w": null}` days; `null` keeps forever) deletes whole month partitions that are entirely older than the cutoff. Retention must exceed every lookback and backtest window.
- **Migration:** `--migrate-cache` imports the milestone-1 CSV cache (`data/cache/<tf>/<ticker>.csv` plus `.from` sidecars) once.
- **Panel:** `build_panel(store, tickers, tf, start, end)` carries open, high, low, close, volume and vwap (NaN where missing).

### 4.4 Offline modes
- **Replay:** run from cached bars only.
- **Synthetic:** a built-in generator (seeded) produces a market of sectors with planted rotations; used when no keys/cache exist and for tests. The UI shows a banner stating the active data mode.

## 5. Flux graph model

Each of the switches A–E below is a `CoreParams` field. Strategy versions (§8.4) record them, so they can be compared side by side. `CoreParams::legacy()` gives the milestone-1 behaviour (dollar pressure, no lift, no inbound pruning, no retention, uniform reference; the liquidity floor and the volume-ratio cap are off). `CoreParams::money_flow()` runs the chain on true dollar flux (dollar pressure, no lift, two-sided pruning, retention) and reads hotness relative to size.

For each bar *t* and stock *i* with return `r_i = C/C⁻ − 1`, volume V and VWAP:

- **(A) Pressure** (`pressure`, default `sqrt`):
  - `dollar`: `p_i = r_i · V_i · VWAP_i`
  - `sqrt`: `p_i = r_i · √(V_i · VWAP_i)`
  - `relative`: `p_i = r_i · V_i / ADV_i`, where ADV_i is the median volume of the previous 20 bars (not including this one); p = 0 until a stock has history. V/ADV is capped at `max_volume_ratio` (default 5; 0 = uncapped), so a single volume spike cannot dominate. This removes the size bias: a stock's pressure reflects how unusual its participation is, not how big it is.
  - Optional volatility scaling (`vol_scale`): returns are divided by the stock's trailing 20-bar volatility (previous bars only, floor 1e-4) before pressure is formed, so low-volatility instruments (bond ETFs) are not structurally cold.
  - `p_i < 0` → net selling (source); `p_i > 0` → net buying (sink). Missing data gives p = 0.
- **Liquidity floor:** a node is active only while its trailing 20-bar median dollar volume is ≥ `min_dollar_volume` (default $1M); computed causally per bar, as part of the active mask.
- **Affinity:** `a_ij = 1 + λ·ρ_ij`, λ ∈ [0, 1], where ρ is the Pearson correlation of returns over the last W = 60 bars. It is computed as a dot product of unit-length centered return vectors u_i, so no N×N matrix is stored. (Milestone 1 used `1 + λ·max(0, ρ)`; the linear form keeps the normalizer exact in O(N·W).)
- **Per-bar flux** for source *i*, sink *j*: `f_ij = |p_i| · p_j · a_ij / D_i` with `D_i = Σ_sinks p_k a_ik = P + λ·u_i·g`, where `P = Σ p_k` and `g = Σ p_k u_k`. Each source distributes exactly its own outflow. Exact row totals `out_i = |p_i|` and column totals `in_j = p_j·(Σ_i a_i + λ·u_j·Σ_i a_i u_i)` (with `a_i = |p_i| / D_i`) are computed in O(N·W).
- **Sparse edges:** each source stores edges only to its top M = 64 sinks by `p_j·a_ij`, chosen from the C = 256 sinks with the largest p_j, with their exact shares. The row and column totals above remain exact.
- **Accumulators:** for each of slow (half-life 20 bars), fast (3) and long-run (120, only when `h_ref = longrun`), per-row sorted edge lists plus decayed `out`/`in` vectors: `F ← 2^(−1/halflife)·F + f`. Each row is capped at 256 edges by weight.
- **(B) Lift** (`lift`, default `excess`): relative to the gravity baseline `E_ij = out_i · in_j / total`, the edge weight is `w_ij = max(0, F_ij − E_ij)` (`excess`), `max(0, F_ij/E_ij − 1)` (`ratio`), or `F_ij` (`off`). This keeps only the structure beyond "big flows meet big flows".
- **(C) Two-sided pruning:** keep the union of each row's top `k_out` = 20 edges and each column's top `k_in` = 10 edges, by w. Ties go to the lower index.
- **(E) Retention** (`retention` ρ_r, default 1.0): row i gives itself the mass `s_i = ρ_r·in_i / (ρ_r·in_i + out_i)` and splits `1 − s_i` over its kept edges in proportion to w. Net buyers therefore hold mass and net sellers pass it on, which makes valleys mean "draining". A row with no kept edges is a self-loop of 1. Inactive nodes (no data in the panel) get no edges in or out.
- **Damping / teleport:** `P' = α·P + (1−α)·(1/N_active)·𝟙` (α = 0.85), solved on the active sub-index.
- **Steady state:** power iteration `π ← π·P'`, warm-started, until ‖Δπ‖₁ < 1e-10 or 1000 iterations.
- **(D) Hotness reference** (`h_ref`, default `uniform`): `h_i = π_i / π_ref,i − 1` with
  - `uniform`: `π_ref = 1/N_active`, so `h = N_active·π − 1`
  - `size`: π_ref ∝ the trailing median dollar volume, so h reads as "hot relative to its size"
  - `longrun`: π_ref is the steady state of the long-run (120-bar) accumulator, so h reads as "hot relative to its own normal"
  - `netflow`: h = (in − out)/(in + out + κ), κ = the median total flow across active nodes. Bounded in (−1, 1); low-flow names shrink toward 0.
- The defaults are confirmed or changed using the evaluation harness (§5.2).

### 5.1 Forecast (no fitted model)
π_t is stationary for P'_slow by construction, so the forecast contrasts two time scales:
- Propagation: `π^(k) = π_t · P'_fast^k` for k ∈ {1, 4, 8} bars, where P'_fast is built from the fast accumulator with the same B/C/E settings.
- Drift: `d_i = π_t,i − π_{t−1},i`.
- **Forecast score:** `s_i = N_active·(π^(k)_i − π_t,i) + β·N_active·d_i` (β = 0.5).

### 5.2 Evaluation harness
`fluxscape --mode replay --eval [--eval-bars B]` runs the pipeline over the cached history (the last B bars, default 120, after a warm-up) for this grid of 15 configurations: legacy; legacy + each of A (relative), B (excess lift), C (k_in = 10), D (size and longrun) and E (retention) on its own; the defaults (all on); defaults with relative pressure; defaults + longrun; defaults + netflow; money-flow; money-flow + netflow; defaults + volscale; and money-flow + volscale. Per configuration it prints:
- **floor share:** the fraction of active nodes whose π is within 1e-6 relative of the teleport floor (1−α)/N_active
- **Gini:** the Gini coefficient of π
- **sector coherence:** the share of off-diagonal raw edge weight between nodes of the same known sector
- **structure gain:** 1 − Spearman(π, inflow share). Near 0 means the chain has collapsed to plain influx (for example a dense, unpruned graph); higher means the pruned multi-hop structure is shaping π.
- **IC:** the mean Spearman correlation between each bar's +1 forecast score and the next bar's return, and its t-statistic; **IC(h):** the same for hotness h, which is where the D settings show up, plus open-to-open variants (decide at close t, fill at open t+1, exit at open t+2), which match the shadow-ledger fill model and are immune to closing bid-ask bounce.
- the mean frame time in ms

### 5.3 Shock mode (counterfactual)
`fluxscape --shock TICKER:SIZE [--shock ...] --top N` runs the pipeline over bars 1..T−2, copies it, then steps the original (baseline) and the copy (shocked) at the last bar T−1. A shock `TICKER:SIZE` adds an extra SIZE% return at the stock's normal volume to the bar's actual pressure, after the liquidity floor and active masking: `p_X += (SIZE/100) × vol_term`, with vol_term = mdv (dollar), √mdv (sqrt) or 1 (relative, applied after the volume-ratio cap), where mdv is the trailing median dollar volume (SIZE < 0 sell-off/source, SIZE > 0 buying surge/sink; duplicate shocks on a node add). Under `vol_scale` the extra return is divided by the node's trailing σ, like a real return: `p_X += (SIZE/100) / max(σ, 1e-4) × vol_term`; a node without return history (fewer than 5 previous returns) has no σ and can't be shocked (an error, like a node without normal volume). Everything else (bar, flux rules, accumulators, transitions, solve) is identical. An unknown or inactive shocked ticker is an error.
The report gives the shocked nodes' Δh and Δπ, the top-N receivers and losers by Δh among active nodes (with Δπ and Δscore(+1)), the total |Δπ| (L1) and the portfolio holdings' Δh.

## 6. Geometry — layout, lattice, landscape (milestone 2)

Built per frame from the active nodes (about 6,000 at N = 10,000 under the $1M floor). Nothing is O(N²) or O(N³).

1. **Territories (user decisions 2026-10-02): flux communities by default, sectors as an option.** Each territory is one contiguous region of the lattice with area proportional to its active stock count, laid out along a generalized Hilbert (gilbert2d) curve (every cell once, consecutive cells at Chebyshev distance 1, starting at (0,0)). `LandscapeParams.territory = flux | sector`; the API accepts `"territory":"flux"|"sector"`.
   - **Flux communities.** The graph is the off-diagonal kept edges of the slow P among active nodes, weighted by `raw` (edges with raw ≤ 0 or non-finite ignored), symmetrized W = R + Rᵀ. Deterministic two-phase Louvain (nodes visited in ascending index, ties to the lowest community id, at most 50 passes per level, aggregated until no change) finds communities of stocks that trade money among themselves. Communities smaller than 8 merge into their most strongly connected neighbour; those with no neighbour pool into one "loose" community that is laid out last. At most 256 communities are kept by merging the smallest.
   - **Order.** The community graph (summed weights) is ordered by recursive spectral bisection: the Fiedler vector of the normalized Laplacian (500 power iterations on 2I − L_sym with the trivial vector deflated), sorted, split where the cumulative stock count is nearest half; disconnected sets fall back to their components. Communities that trade with each other therefore sit next to each other.
   - **Stability.** The clustering is recomputed on the first frame, every 5 frames and whenever parameters change. In between, membership and order are kept; a newly active stock joins its strongest active neighbour's community (else loose) and an inactive stock leaves. On a re-cluster, new communities are matched to the old ones greedily by overlap (Jaccard ≥ 0.3) so ids and relative order persist.
   - **Sectors.** In sector mode a node's group is its `Security::sector` (ids in ascending string order); a sector with no active stocks has no territory.
   - **Sizes.** With A active stocks and C lattice cells, territory g gets c_g = a_g plus its largest-remainder share of the C − A spare cells (ties to the lower id). Territories are consecutive ranges of the curve in ascending group order.
   - **Inside a territory,** cells are ordered by squared distance to the territory centre (ties by cell index). Stocks are ranked by smoothed hotness s = (1−β)·signed-log h + β·previous s (β = 0.5; non-finite h ranks as 0). A territory is a mountain when its median s is at or above the median over all active stocks, and a crater otherwise: a mountain places the highest s nearest the centre, a crater the lowest. The farthest c_g − a_g cells stay empty and IDW fills them, which forms seams between territories.
   - Every vertex still shows its stock's exact value. Cost is O(C log C) for placement and O(E log n) for Louvain (about 150 ms at 10,000 nodes of degree 30, only on re-clusters).
2. **Lattice.**
   - The lattice is cols = ⌈√N_active⌉ by rows = ⌈N_active/cols⌉. A node's cell is row·cols + col.
   - Placement is recomputed each frame. Stability comes from the smoothed ranking, not from cell hysteresis. (The recursive-bisection snap and hysteresis helpers remain in `lattice.*` but are unused.)
3. **IDW landscape.**
   - The raster has (cols·s) × (rows·s) pixels, with s = 4 by default. Each pixel is `z = Σ wᵢhᵢ / Σ wᵢ`, wᵢ = 1/dᵢ^q (q = 2), over occupied lattice cells within radius R = 3 cells.
   - If no cell is within R, the nearest occupied cell is used. A pixel exactly on a node takes that node's value.
   - Computed in parallel per pixel row.
4. **Height.** The default height is `signed-log`: `sign(h)·log1p(|h|)`, because money-flow hotness spans orders of magnitude. `linear` is also available, and suits bounded `netflow` hotness. The colour scale is diverging (valleys blue, hills red), symmetric around 0.
5. **Arcs.** The global top 2,000 off-diagonal raw-flux edges between active nodes.

Graph adjacency ≠ geometric adjacency: the layout approximates flux proximity in 2D, and the arcs show the real links.

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

### 7.2b Optional: simplex-weighted, walk-forward signal blend (gated)
Built only if the evidence supports it. The gate is that at least two configurations show **independent, stable out-of-sample signal**: open-to-open IC with |t| ≥ 3 in two or more non-overlapping walk-forward windows, and a pairwise rank correlation of their scores below 0.5.
- Inputs: K per-stock signals per bar (hotness or forecast scores of the qualifying configurations).
- Blend: `s = Σ w_k · rank_k`, with weights `w ≥ 0, Σ w = 1`, chosen to maximize mean rank IC on a training window (projected gradient on the simplex; no neural network).
- Walk-forward: fit on window t, apply unchanged to window t+1, and report only out-of-sample results. The refit cadence equals the window length.
- Lives as its own strategy version (§8.4), so the shadow ledger compares it with its best single component in real P&L. It is dropped if it doesn't beat that component out of sample.
- This is a fitted layer on top of the no-fitted-model engine, and is labelled as such.

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

### 9.2b Serve mode (milestone 2)
`fluxscape --serve [--port 8765] [--web web] [--mode replay|alpaca|synthetic] [model flags]`. The default preset is **money-flow** unless `--legacy`, `--money-flow` or explicit model flags change it.
- A background thread computes core frames and landscape frames for the data window and keeps up to 300 landscapes in memory for the scrubber.
- Before the last bar it keeps a copy of the pipeline, so shocks at the latest bar cost about two frames.
- A parameter change cancels the computation and restarts it.

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
| `GET /api/status`, `GET /api/times` | computation progress and parameters; frame times (milestone 2) |
| `POST /api/shock`, `GET /api/shock/grid` | counterfactual shocks at the latest bar (§5.3): deltas plus a Δh raster (milestone 2) |

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

Milestone 1.5 additions:
- ReturnWindow correlation equals brute-force Pearson, including after ring wraps.
- Sparse flux: with C = M = N it equals the dense formula; Σ in = Σ out; the column totals are exact even when edges are truncated.
- Lift: a rank-one (gravity-only) flux gives no edges; a planted above-gravity edge survives. `k_in` keeps a node's inbound edges even when no row selects them. Retention gives P_ii = in/(in+out).
- Evaluation metrics (Gini, Spearman, floor share, sector coherence) are checked against hand-computed values. On a heavy-tailed synthetic market the defaults have a lower floor share than legacy.
- Benchmark (skipped by default): a 10,000-node synthetic daily frame takes under 1 s in Release.

Integration:
- A synthetic market with a planted rotation (sector A → sector B) makes sector B the top hill, places it in a contiguous lattice patch, and makes the daily strategy book rotate into B and beat the baseline in that synthetic run.
- End-to-end server smoke test: start in synthetic mode, fetch frame JSON, grid and books.

## 12. Implementation milestones

1.6. **Storage:** DuckDB + Hive-partitioned Parquet lake (spec §4.3), per-batch commits, compaction, retention, CSV migration, windowed panel with OHLC; then the first live 10K run and its evaluation.
1.5. **Model A–E + 10K scale (CLI):** sparse flux, pressure modes, lift, two-sided pruning, retention, hotness references, Alpaca assets universe with liquidity ranking and rate limiting, heavy-tailed synthetic market, evaluation harness, 10,000-node benchmark. Status: replaces the milestone-1 dense flux builder.
1. **Core model (CLI):** universe, synthetic market, Alpaca client and cache, flux → solver → forecast; CLI prints top hills and valleys.
2. **Geometry + server + UI:** layout, lattice, IDW, REST/SSE, deck.gl landscape and left-panel controls.
3. **Optimizer + ledger:** ADMM, shadow books, strategy versions, backtest, catch-up, performance panel.

## 13. Caveats

- The flux is inferred from simultaneous price and dollar-volume moves; it is not observed order flow.
- Hotness-to-return is not a calibrated relationship; the shadow ledger exists precisely to measure whether it works. Past shadow performance does not imply future results.
- The bundled S&P 500 list may be out of date.
- The free Alpaca plan means SIP data runs about 15 minutes behind the market.
