# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MarketRank is a C++20 engine that estimates stock-to-stock money flows from daily bars, builds a sparse Markov
chain over them and ranks stocks by its stationary distribution π. Fluxscape is the 3D landscape web UI over the
result. The README is the long-form spec (model, landscape algorithms, walk-forward, 13F); read the relevant
section before changing behaviour it documents, and keep it in sync.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j   # deps (json, httplib, doctest, DuckDB) via FetchContent
./build/marketrank_tests                                   # all C++ tests (doctest)
./build/marketrank_tests -tc="*stationary*"                # one test case by name pattern
./build/marketrank_tests -sf="*test_markov_solver*"        # one test file
scripts/ui_smoke.sh                                        # headless-Chrome UI smoke test + screenshot (MODE=replay for real data)
python3 -m unittest discover scripts/tests                 # Python helper tests (13F / OpenFIGI)
research/.venv/bin/python -m unittest discover research/graphlearn/tests   # research (torch) tests
paper/build.sh                                             # paper/marketrank.pdf
```

Sources are globbed (`src/**/*.cpp` → `marketrank_core` static lib, `tests/*.cpp` → `marketrank_tests`), so new
files only need a re-run of cmake. `marketrank_tests` depends on the `marketrank` binary (the serve-lock test runs it).

## Run

```bash
./build/marketrank --serve --mode synthetic        # UI on http://127.0.0.1:8765, no keys needed
./build/marketrank --mode alpaca --universe-size 10000   # sync daily bars into data/lake (Alpaca keys in .env)
./build/marketrank --serve --mode replay           # UI over cached data
./build/marketrank --mode replay --walkforward --lookback-days 3750   # walk-forward backtest → data/walkforward/<run_id>/
./build/marketrank --help                          # every switch
```

Modes: `synthetic` (generated market, no network), `alpaca` (fetch + solve), `replay` (cached lake only).

## Architecture

Pipeline per bar (`src/pipeline/core_pipeline.*`, parameters in `CoreParams`):
universe → pressure (`graph/pressure`: r·V·VWAP etc.) → sparse flux (`graph/sparse_flux`: sources split outflow over
sinks, correlation-tilted, top edges only) → exponentially decayed slow/fast/cluster memories → transition matrix
CSR (`graph/transition`, `graph/csr`) → damped power iteration for π (`graph/markov_solver`) → hotness and
forecasts (`graph/hotness`, `graph/forecaster`).

- `market/` — data sources: Alpaca client, synthetic market, universe/snapshot selection, SEC sectors, session
  calendar, intraday, and `Panel` (the bars matrix every stage consumes).
- `storage/` — DuckDB-managed, Hive-partitioned Parquet lake in `data/lake` (month partitions, retention via `--maintain`).
- `geom/` — Fluxscape layout: Louvain communities, spectral order + Gilbert curve territories, lattice, IDW,
  CVT smoothing. Display only; never feeds back into π.
- `server/` — cpp-httplib server serving `web/` (plain `index.html`/`app.js`/`style.css`, no build step) and
  `/api/*`; `FrameStore` caches per-bar `LandscapeFrame`s (plus an alternate set with ETFs flipped).
- `walkforward/` — causal weekly/monthly backtest: signals, gated blend, backtest, stats/DSR gate, report, and
  `registry.csv` of every trial. `external.*` scores signals produced outside the engine (`--wf-external`).
- `flows13f/` + `scripts/sec13f.py`, `scripts/openfigi_map.py` — validation of estimated flows against SEC 13F
  holdings changes.
- `analysis/` — cluster persistence and partition metrics. `cli/` — argument parsing, rank report, panel export.
- `research/graphlearn/` — Python/PyTorch experiments (M4–M6) with pre-registrations (`PREREGISTRATION.md`,
  `POWER*.md`, `REAL_M6.md`) and `frozen_settings.json`. All three were negative; results are recorded there and in
  the paper.

## Invariants and conventions

- Results are bit-identical for any `--threads N` (OpenMP); keep reductions deterministic.
- The walk-forward is strictly causal (decisions use data up to the rebalance date, trades at next open, one-period
  embargo on the blend). Re-running a run id with different parameters is an error by design.
- Pre-registered experiments: settings are frozen before real-data runs; don't tune on real results.
- The web UI has no model/landscape knobs (only the "Show ETF clusters" checkbox). Developer knobs belong on the
  CLI and `POST /api/params`, not the UI.
- The landscape is a reading aid; the exact sorted π table (`/api/top`, "MarketRank top 10") is the source of truth.
- Commit directly to `master`. The repo is public (github.com/karametebkaan/marketrank): never commit `.env` or
  `data/` contents (lake, caches, runs, 13F downloads).
