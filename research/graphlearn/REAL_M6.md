# M6: MarketRank's chain on real 15-minute data (pre-registered)

This file and `real_m6.py` were committed **before** any real 15-minute data was trained on or evaluated. The
rule below is the one `real_m6.verdict` applies; the git history of this file is the record.

## Why

M5 found that a learned correction to the chain is not recoverable at realistic signal levels
(`POWER_M5.md`). The open question is the chain itself: does MarketRank's transition graph, as it is, carry
predictive information intraday? The Markov transition graph stays the core. Nothing
is learned about the graph in M6.

## Data

The M5 data scope approved on 2026-10-03, unchanged:
- 15-minute regular-session bars, 2024-10-01 to 2026-10-02;
- the static top-1000 universe `data/universe/intraday_top1000.csv` (survivorship bias acknowledged);
- the panel from `marketrank --mode replay --export-panel DIR --timeframe 15m`, with label_6 (the 6-bar forward
  return, masked across the close) and, at every bar, MarketRank's kept transition edges computed from bars up to
  that bar.

**Data gate** (checked before training; if it fails, nothing is trained and that is reported): at least 400
sessions, and a median of at least 800 eligible names per labeled bar.

## Variants

All share the encoder, head and training protocol of `wf_m5.DEFAULTS` (expanding window, monthly retrain,
first prediction at session 60, embargo 1 session, validation-IC early stopping, 2 restarts). They differ only in
the message fed to the head:
- **Bprior:** the message is averaged over MarketRank's chain at that bar, A = P_prior.
- **Bshuf:** the same chain with every edge's destination ticker relabeled by one fixed random permutation
  (`wf_m5.SHUF_SEED = 20261004`). Same graph shape and weights, wrong tickers. This controls for "any neighbour
  average helps".
- **B0E:** a per-stock embedding, no message.
- **B0:** no message.

On synthetic markets where the chain is the truth, Bprior beats Bshuf (t = 6.95); where nothing is predictive, it
does not (t = −1.24) (`tests/test_m5.py`).

## Runs

- **Primary:** rolling mode (the deployed regime), model seeds 1 and 2.
- **Secondary:** scratch mode, model seed 1. Reported, not gated.

## Pre-registered rule

For **each** primary run (rolling, seed 1 and seed 2), both must hold:
1. **Bprior − B0E:** paired per-bar IC difference with day-clustered t > 2 (each session's mean difference is one
   unit), **and** a positive mean difference in at least 2/3 of calendar months.
2. **Bprior − Bshuf:** day-clustered t > 2.

**Pass** = both conditions on both primary runs. Anything else is a fail, reported as such.

**Reported, not gated:** Bprior − B0, Bshuf − B0E, the scratch run, and a top-minus-bottom decile spread of label_6
on non-overlapping bars (session slots 0, 6, 12, 18), gross and net of 5 bps per side on both legs (20 bps per
period).

**One shot.** The rule, the variants, the protocol and the data scope are fixed here. If the code must change after
the real run starts (a crash, a data bug), the change and its reason are recorded in this file before rerunning, and
no setting is changed in response to the results.

## Change log (after the rule was committed)

- **2026-10-04, before any training, data gate not yet computed.** `real_m6.py` crashed loading the real export:
  the C++ `--export-panel` writes `session` as an object (the per-bar list is `session.index`) and puts the prior
  files under `prior.dir`, and the Python reader expected a bare list and no subdirectory. M5 never read a real
  export, so this had not come up. Fixed in `intraday._resolve_layout` (reader only), with a test in the export's
  shape. No rule, variant, protocol or data setting changed.
- **Noted, not changed:** the export defines label_6 one bar later in each session than `intraday.label_mask`
  (bars whose t+1..t+6 are in the session, against t+1..t+7). The frozen protocol keeps the stricter Python mask, so
  19 of the 20 labelable bars per session are used. This is conservative and cannot leak future data.
- **Data gate (computed after the fix):** 503 sessions, median 964 eligible names per labeled bar. Passes.
- **2026-10-04, first real run killed for low memory (no results read).** Each run cached every session batch for
  its whole life: 27 MB per session on the real panel (~692,000 edges), ~25 GB per variant, and Bshuf held a second
  full cache. Three runs in parallel ran the machine out of memory, and Claude Code stopped them. Nothing had been
  evaluated: rolling runs were partway through Bshuf, the scratch run through Bprior. Fix (`wf_m5.main`): each
  variant's cache is freed when it finishes, and Bshuf's is built only for Bshuf; training and predictions are
  unchanged. `real_m6.py` now runs 2 workers, scratch first (scheduling only). The code digest changed, so the runs
  restart fresh; the killed run is kept at `data/m6_runs_killed_2026-10-04/` and not used.

## Results

The run finished on 2026-10-05 (`real_m6_results/`: `results.md`, `results.json`, one `run_*.json` per run).
**The pre-registered rule fails.** MarketRank's chain does not predict 6-bar returns better than the no-graph
baseline, and it does no better than the same chain on the wrong tickers.

| primary run | Bprior − B0E (t) | months positive | Bprior − Bshuf (t) | pass |
|---|---|---|---|---|
| rolling, seed 1 | −0.0014 (−0.5) | 39% | +0.0001 (+0.2) | no |
| rolling, seed 2 | −0.0011 (−0.5) | 39% | +0.0006 (+0.9) | no |

**What this says:**
- **Every variant has the same small IC,** about +0.008 to +0.010 (t ≈ 2.5–3.8). It comes from the shared node
  features (recent returns, volume shock, volatility, pressure), not from any graph. B0, with no message at all, is
  in the same range.
- **The chain adds nothing on top.** Bprior − B0E is slightly negative in both rolling runs and the scratch run
  (t between −0.5 and +0.4), positive in only about 39% of months. Bprior − Bshuf is indistinguishable from zero
  (t ≤ 0.9): the identity of who sends to whom carries no measurable 6-bar information here.
- **The scratch run agrees** (Bprior − B0E t = +0.4, Bprior − Bshuf t = −0.1).
- **Decile spread (reported only):** negative for every variant, including the graph-free ones (gross −0.02% to
  −0.06% per 1.5 h, t ≈ −1.3 to −2.9), and about −0.22% to −0.26% net of costs. A positive rank IC with a negative
  extreme-decile spread means the extremes of the score behave differently from the bulk; since it holds for B0 too,
  it is a property of the shared features, not of the chain.

M6 stops here, as pre-registered. On this test, the intraday chain as built does not earn an intraday product view;
the daily MarketRank (ranking and Fluxscape) is unchanged.
