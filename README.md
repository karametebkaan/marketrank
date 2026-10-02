# MarketRank

MarketRank ranks stocks by the stationary distribution of a Markov chain over stock-to-stock money flows, à la PageRank; Fluxscape is its 3D landscape view.

Models the market as a flux graph (money leaving net-sold stocks for net-bought ones),
solves its Markov steady state like PageRank, and ranks the hottest and coldest stocks.
Design: `docs/superpowers/specs/2026-10-01-marketrank-design.md`.

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
./build/marketrank --serve --mode replay            # http://127.0.0.1:8765 (money-flow preset)
./build/marketrank --serve --mode synthetic --port 9000
scripts/ui_smoke.sh                                # headless-Chrome smoke test + screenshot
```

The page shows a triangulated landscape of signed-log hotness. Hills are where money settles relative to size, and valleys are where it drains.

**Placement.** Territories are flux communities by default: Louvain on the model's own flux graph finds stocks that trade money among themselves, and spectral bisection of the community graph orders the communities so that trading partners sit next to each other along a Hilbert-type curve. Each community owns one contiguous region with area proportional to its stock count, and inside it stocks are ordered by hotness from the centre outward along a spiral, so a community reads as a mountain (inflow) or a crater (outflow). Choose "sectors" in the Territories select (or `"territory":"sector"` in `/api/params`) to use market sectors instead.

**Stability.** Stocks stay put from bar to bar. Inside a territory the cells are taken in spiral order (ring by ring, each ring by angle), so a small change in rank is a small move. On top of that a stock keeps its cell while it stays in the same community, its cell is still inside the territory, and its new place in the spiral is within 15% of the territory (at least 2 slots) of that cell. The clustering is refreshed every 5 bars. Each refresh starts Louvain from the current communities rather than from scratch, and the new communities are matched to the old ones (Jaccard ≥ 0.3, or ≥ 60% of the new community inside an old one), so community labels and their order persist. The first 5 bars only warm up the model. On the real replay data about 85% of stocks keep their cell between refreshes, and about 74% keep their community at a refresh.

**Surface.** The mesh vertices are the lattice points. Each stock's value (`hdisp`, its signed-log hotness) is exact; empty vertices are filled by IDW, and the displayed surface is then Gaussian-smoothed with σ = 1 cell by default (the Smoothing slider; edge windows are renormalized), so a vertex shows a local average rather than the stock's exact value. The shock Δh surface is smoothed the same way. The tooltip and the Hottest-10 table always show the exact values. Each triangle is Gouraud-shaded from its vertex colours, red for inflow and blue for outflow, with relief lighting on top. Changing only the display settings (smoothing, height, IDW, subdivision) redraws the cached frames without recomputing the model.

**Overlays.** The strongest flux arcs and your portfolio, as green rings, are drawn on top.

**Left panel.** It lets you:
- switch the preset and hotness reference;
- choose the territories (flux communities or sectors);
- set the IDW and height parameters and the display smoothing;
- set the arc count with the Arcs slider;
- scrub or play through the last 300 bars (the Hottest-10 table at the top right shows the ten highest model h with their recent history);
- apply a shock on the latest bar (`TICKER`, ±%) to show the Δh landscape of who absorbs the money and who loses it.

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
