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
./build/fluxscape --mode synthetic                 # no keys needed
cp .env.example .env                               # add Alpaca keys
./build/fluxscape --mode alpaca --timeframe 1d     # fetch and cache, then solve
./build/fluxscape --mode replay --timeframe 1d     # cached data only
python3 scripts/fetch_sp500.py                     # refresh the S&P 500 list
```

Advisory and experimental. The flux is inferred from price and dollar-volume
co-movement, not observed order flow.
