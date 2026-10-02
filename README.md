# MarketRank

MarketRank ranks stocks by the stationary distribution of a Markov chain over stock-to-stock money flows, à la PageRank; Fluxscape is its 3D landscape view.

Models the market as a flux graph (money leaving net-sold stocks for net-bought ones),
solves its Markov steady state like PageRank, and ranks stocks by their MarketRank score π·N (and by hotness).
Design: `docs/superpowers/specs/2026-10-01-marketrank-design.md`.

## MarketRank model

The default model (`CoreParams::market_rank()`, `--marketrank`) ranks stocks by the stationary distribution π of a
damped Markov chain over the paired stock-to-stock dollar flows T, using the MarketRank Condition (MRC):

```
r_i = (1 − p) · Σ_{j ∈ B(i)} P_i(r_j) + p / N,   P_i(r_j) = r_j · T_{j→i} / Σ_k T_{j→k},   p = 0.15
```

B(i) is the set of stocks sending money to i, so each stock passes its rank on in proportion to the out-share
of its dollar flow, and p/N is the teleport (α = 1 − p = 0.85). T is the dollar flux accumulated over the whole
data window (slow half-life 1e9 bars), with no lift, no self-retention and the default two-sided pruning
(top 20 out-edges per row, top 10 in-edges per column). A stock that passes nothing on (an empty row) is a
dangling node: its rank teleports uniformly, as in PageRank.

The **MarketRank score** is π_i·N_active (1 = average) and the **heartbeat** is its pulse,
Δlog(π_i·N_active) between the current bar and the previous one (so a change in the number of active stocks
alone is no pulse). Under this preset the forecast column "score+1" contrasts the cumulative π with the 3-bar
fast chain, so it reads "where the newest flow points against the whole window", not a one-bar drift.

Worked example (three stocks, T in dollars):

| From → To | A | B | C |
|---|---|---|---|
| A | – | 500K | 100K |
| B | 200K | – | 1M |
| C | 750K | 400K | – |

Row-normalizing T gives P; the damped power iteration from the uniform start gives, after 7 iterations,
B 0.35980, C 0.34770, A 0.29248, and converges to B 0.36016, C 0.34665, A 0.29319. B ranks first: it
receives the most money relative to what its senders pass on. The test suite checks both results through the
solver and through the pipeline's transition builder.

`./build/marketrank --mode replay` prints the TOP and BOTTOM MARKETRANK tables (rank, ticker, sector,
MarketRank π·N, heartbeat Δlog(π·N), hotness h, score+1), sorted by π with ties to the lower index, the same
order as `/api/top`. The stocks that receive no flow all tie at the teleport floor; the bottom table folds them
into one line ("N stocks tied at the teleport floor (π·N = x)") and lists the lowest names above it.
`--rank-by hotness` sorts by h instead (HILLS and VALLEYS). `--money-flow`, `--legacy` and `--defaults`
(`CoreParams{}`) select the other presets.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/marketrank_tests
```

## Run

```bash
./build/marketrank --mode synthetic                          # no keys needed
cp .env.example .env                                        # add Alpaca keys
./build/marketrank --mode alpaca --universe-size 10000       # build/reuse 10K liquidity snapshot, sync, solve
./build/marketrank --mode replay                             # cached data, latest snapshot (or --universe sp500)
./build/marketrank --mode replay --eval --eval-bars 120      # compare A-E settings (floor, Gini, coherence, IC)
./build/marketrank --mode replay --rank-by hotness --money-flow  # size-relative hotness, the earlier serve default
./build/marketrank --mode replay --legacy                    # milestone-1 behaviour
./build/marketrank --mode replay --shock NVDA:-10 --top 8       # counterfactual: add an extra -10% return to NVDA at the last bar, see who absorbs it
./build/marketrank --help                                    # all switches (--pressure, --lift, --k-in, ...)
./build/marketrank --migrate-cache                           # one-time: import the old data/cache CSVs into data/lake
./build/marketrank --maintain                                # compact partitions and apply data/lake/retention.json
./build/marketrank --mode alpaca --refetch-full             # one-time: refetch every ticker's full stored history (repairs old split/dividend bases)
./build/marketrank --sync-sectors --universe snapshot  # fill sectors from SEC EDGAR SIC codes (run on its own)
python3 scripts/fetch_sp500.py                              # refresh the S&P 500 list
```

`--threads N` sets OpenMP threads; results are bit-identical for any thread count.
`data/universe/include.csv` / `exclude.csv` (one ticker per line) override the universe filters.

Sectors: only S&P 500 names have a GICS sector in the snapshot. `--sync-sectors` fetches SEC EDGAR SIC
codes for every ticker of the newest snapshot, as rank loads it in replay mode (honours `--universe`; `--universe-size` is ignored, the cache is keyed by ticker), into
`data/sectors/sec_sic.csv` and prints per-sector counts and the % Unclassified before and after. It
needs `SEC_USER_AGENT="Your Name your@email"` in `.env` (SEC requires a contact), no Alpaca keys, and
does not touch the lake. It is limited to 8 requests/s, caches for 90 days and resumes if interrupted;
run it separately from `--mode alpaca`. At load time every sector is filled in the order S&P GICS,
SEC SIC, then `ETF/Fund` (name heuristic and `funds.csv`), then `Unclassified`; snapshot CSVs are unchanged.

Advisory and experimental. The flux is inferred from price and volume co-movement,
not observed order flow.

## Landscape UI (Fluxscape)

```bash
./build/marketrank --serve --mode replay            # http://127.0.0.1:8765 (marketrank preset)
./build/marketrank --serve --mode synthetic --port 9000
scripts/ui_smoke.sh                                # headless-Chrome smoke test + screenshot
```

The page shows a triangulated landscape of **π relative to size**: the height is log(π_i / s_i), where s_i is the stock's size share (trailing median dollar volume over the sum across active stocks, the size `HotRef::Size` uses). A hill attracts more money than its size predicts, a valley less, and 0 is exactly as predicted; stocks without a size get no height. Ranking and the table show π itself: the landscape is a reading aid, the table is the truth. Note that the many stocks at the teleport floor get the same π whatever their size, so small floor stocks read slightly above 0 here. The same value orders the stocks inside each territory and decides mountain or crater. The legend under the landscape names the quantity.

**The UI is fixed.** It has no model or landscape options: the marketrank preset, π relative to size, flux territories, the CVT smoother and subdivision 1 are fixed defaults, and the page never calls `/api/params`. Developer knobs exist only on the CLI (`--money-flow`, `--h-ref`, ...) and in the API (`POST /api/params`: `value` `pi_rel_size|pi|hotness`, `territory`, `smoother`, IDW, ...).

**Placement.** Territories are flux communities by default: Louvain on the model's own flux graph finds stocks that trade money among themselves, and spectral bisection of the community graph orders the communities so that trading partners sit next to each other along a Hilbert-type curve. Each community owns one contiguous region with area proportional to its stock count, and inside it stocks are ordered by hotness from the centre outward along a spiral, so a community reads as a mountain (inflow) or a crater (outflow). `"territory":"sector"` in `/api/params` uses market sectors instead (developer option).

**Stability.** Stocks stay put from bar to bar. Inside a territory the cells are taken in spiral order (ring by ring, each ring by angle), so a small change in rank is a small move. On top of that a stock keeps its cell while it stays in the same community, its cell is still inside the territory, and its new place in the spiral is within 15% of the territory (at least 2 slots) of that cell. The clustering is refreshed every 5 bars. Each refresh starts Louvain from the current communities rather than from scratch, and the new communities are matched to the old ones (Jaccard ≥ 0.3, or ≥ 60% of the new community inside an old one), so community labels and their order persist. The first 5 bars only warm up the model. On the real replay data about 85% of stocks keep their cell between refreshes, and about 74% keep their community at a refresh.

**Surface.** The mesh vertices are the lattice points. Each stock's value (`hdisp`: log(π / size share); log(π·N) or signed-log hotness through the API) is exact; empty vertices are filled by IDW, and the displayed surface is then smoothed for the eye, so a vertex shows a local average rather than the stock's exact value. The default CVT smoother is a Lloyd-style relaxation over each pixel's 3×3 neighbourhood weighted by density ρ = ε + |z|: tall neighbours pull the pixel toward them, which fills the dip that IDW leaves around an isolated peak (12 iterations, λ = 0.6, ε = 0.1·P90|z|; the API also offers Gaussian σ and none). It does not conserve the mean height. The shock Δh surface is smoothed the same way. The landscape is aesthetic and the sorted table is the truth: the tooltip, the top-10 table and `/api/top` always show the exact, unsmoothed values. Each triangle is Gouraud-shaded from its vertex colours, red for inflow and blue for outflow, with relief lighting on top. Changing only the display settings (smoother, height, IDW, subdivision) redraws the cached frames without recomputing the model.

**Overlays.** The strongest flux arcs and your portfolio, as green rings, are drawn on top.

**Left panel.** The status line reads "MarketRank · replay 1d · N stocks · date". The panel lets you:
- scrub or play through the last 300 bars, or follow the latest;
- set the height scale (visual only), the arc count and the labels;
- see your portfolio's π·N;
- apply a shock on the latest bar (`TICKER`, ±%) to show the Δh landscape of who absorbs the money and who loses it.

The table at the top right, "MarketRank top 10 · π (exact)", shows the ten highest π·N with their heartbeat and recent history. Tooltips show π·N, the heartbeat, π, h and the displayed height value.

## How it works

1. **Universe.** Start from every tradable US stock on Alpaca. Drop warrants, units, rights and
   ETF-like products. Rank what's left by median daily dollar volume and keep the top N (default
   10,000). Your portfolio holdings are always included.
2. **Buying and selling pressure.** For each bar, each stock's pressure is its return times the
   square root of its dollar volume (`r · √(V · VWAP)`, the default). Positive pressure means net
   buying (a sink); negative means net selling (a source). `--pressure relative` instead uses how
   unusual volume is against its own normal (`r · V / ADV`, with V/ADV capped at 5); `--vol-scale` additionally divides each return by the stock's trailing
   20-bar volatility so calm instruments are not structurally cold. Stocks whose
   median dollar volume is under $1M stay out of the graph.
3. **Money flux, with no fitted model.** Each source's outflow is split across the sinks in
   proportion to their pressure, and tilted toward stocks it moves with (rolling return
   correlation). The result is a directed, weighted graph: edge i→j is the estimated money moving
   from selling i into buying j. Only the strongest edges are stored, so memory grows with N, not N².
4. **Memory of the flow.** Edges and totals build up with exponential decay: a slow memory
   (half-life 20 bars) for the equilibrium and a fast memory (3 bars) for the newest flow. Bars are stored in a DuckDB-managed,
   Hive-partitioned Parquet lake (`data/lake`), with month partitions and a retention policy.
5. **Keep the real structure.** Subtract the flow you'd expect from size alone ("big buyers meet
   big sellers"). Keep each stock's strongest outgoing and incoming edges. Let net buyers retain
   part of what flows in, so money collects where it's being bought.
6. **Markov chain → steady state.** Normalize each stock's edges into transition probabilities
   and solve for the stationary distribution π, the PageRank-style steady state. π is where the
   market's money settles if the current flow pattern continues.
7. **Hotness.** Hotness `h = π / reference − 1`: hills (h > 0) are where money accumulates and
   valleys (h < 0) are where it drains. The reference can be uniform, size, each stock's own
   long-run normal, or a bounded net-flow ratio `(in − out)/(in + out + κ)`.
8. **Forecast.** Push the steady state one, four and eight bars forward through the fast flow,
   and add its recent drift. The result is a score for where the money is heading next.
9. **Evaluate before trusting.** `--eval` compares every setting on historical data. It reports
   how much of the graph is dead, how concentrated π is, whether flows cluster by sector, and
   whether the scores rank next-bar returns (information coefficient).
10. **Next milestones.** The steady state becomes a 3D landscape: stocks are laid out by flux and
    snapped to a grid, with inverse-distance-weighted terrain drawn in deck.gl. Then the portfolio
    optimizer moves your holdings "uphill", and the hourly, daily and weekly suggestions are
    paper-traded so their real profit and loss is tracked.

Shock mode (`--shock TICKER:SIZE`) replays the last bar twice, once as is and once with an extra SIZE% return, at normal volume, added to the named stocks' pressure, and reports which stocks gain or lose hotness.
