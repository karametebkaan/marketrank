# Fluxscape

Models the market as a flux graph (money leaving net-sold stocks for net-bought ones),
solves its Markov steady state like PageRank, and ranks the hottest and coldest stocks.
Design: `docs/superpowers/specs/2026-10-01-fluxscape-design.md`.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fluxtests
```

## Run

```bash
./build/fluxscape --mode synthetic                          # no keys needed
cp .env.example .env                                        # add Alpaca keys
./build/fluxscape --mode alpaca --universe-size 10000       # build/reuse 10K liquidity snapshot, sync, solve
./build/fluxscape --mode replay                             # cached data, latest snapshot (or --universe sp500)
./build/fluxscape --mode replay --eval --eval-bars 120      # compare A-E settings (floor, Gini, coherence, IC)
./build/fluxscape --mode replay --legacy                    # milestone-1 behaviour
./build/fluxscape --help                                    # all switches (--pressure, --lift, --k-in, ...)
./build/fluxscape --migrate-cache                           # one-time: import the old data/cache CSVs into data/lake
./build/fluxscape --maintain                                # compact partitions and apply data/lake/retention.json
python3 scripts/fetch_sp500.py                              # refresh the S&P 500 list
```

`--threads N` sets OpenMP threads; results are bit-identical for any thread count.
`data/universe/include.csv` / `exclude.csv` (one ticker per line) override the universe filters.

Advisory and experimental. The flux is inferred from price and volume co-movement,
not observed order flow.

## How it works

1. **Universe.** Start from every tradable US stock on Alpaca. Drop warrants, units, rights and
   ETF-like products. Rank what's left by median daily dollar volume and keep the top N (default
   10,000). Your portfolio holdings are always included.
2. **Buying and selling pressure.** For each bar, each stock's pressure is its return times the
   square root of its dollar volume (`r · √(V · VWAP)`, the default). Positive pressure means net
   buying (a sink); negative means net selling (a source). `--pressure relative` instead uses how
   unusual volume is against its own normal (`r · V / ADV`, with V/ADV capped at 5). Stocks whose
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
   valleys (h < 0) are where it drains. The reference can be uniform, size, or each stock's own
   long-run normal.
8. **Forecast.** Push the steady state one, four and eight bars forward through the fast flow,
   and add its recent drift. The result is a score for where the money is heading next.
9. **Evaluate before trusting.** `--eval` compares every setting on historical data. It reports
   how much of the graph is dead, how concentrated π is, whether flows cluster by sector, and
   whether the scores rank next-bar returns (information coefficient).
10. **Next milestones.** The steady state becomes a 3D landscape: stocks are laid out by flux and
    snapped to a grid, with inverse-distance-weighted terrain drawn in deck.gl. Then the portfolio
    optimizer moves your holdings "uphill", and the hourly, daily and weekly suggestions are
    paper-traded so their real profit and loss is tracked.
