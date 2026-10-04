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

## Results

(filled in after the run; see `real_m6_results/`)
