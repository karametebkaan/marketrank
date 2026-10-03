# M4 pre-registration: does a learned flow graph predict next-week returns?

Written 2026-10-03, **before** any real-data model run counts. The only real-data contact so far was one plumbing
smoke run (2 epochs) during development. It is registered here as trial 0 and never used to choose settings.

## Question
Can a stock-to-stock connectivity matrix A, learned end to end with a small neural net, improve the out-of-sample
forecast of next-week open-to-open returns beyond the identical net without a graph?

## Variants (identical encoder, head, seeds, epochs, dates and node sets)
- **learned:** low-rank, top-k learned A, with a message passed along A.
- **B0:** no message. This is the decisive baseline.
- **B0E:** B0 plus each stock's own learned embedding, with no message. This controls for "ticker identity" posing as a graph.
- **B2:** a sector-block A.
- **B1:** MarketRank's estimated P, if supplied.

## Settings
Hyper-parameters are chosen **only on synthetic planted and null markets**, using the selection rule written in
POWER.md before that grid was run. They are frozen in `frozen_settings.json` before the real run.

## Endpoints
- **Primary:** `--mode rolling` (the regime that would be deployed), retraining every 13 weekly rebalances, embargo 1 week.
- **Robustness:** `--mode scratch` with the same settings.
- **Evaluation:** the M3a walk-forward harness via `--wf-external`, over each external's scored span only. Every variant, mode and seed run is a registry trial.

## Success: all must hold
1. **Paired IC vs B0:** learned − B0 per rebalance has t > 2 and is positive in at least 6 of the scored years.
2. **Paired IC vs B0E:** learned − B0E has t > 2, so the gain is not ticker identity.
3. **Tilt:** learned's tilt beats the rebalanced base after 10 bps (paired weekly excess vs B0 > 0).
4. **Gate:** learned passes the M3a gate c1–c3 over its scored span, with DSR_excess deflated by the total registered trial count.
5. **Robustness:** scratch shows the same sign for learned − B0 with t > 1. If it disagrees, the result is reported as "not robust".

Anything else is reported as a negative result. A negative result is read as **"no learnable connectivity above the
power-curve detection threshold"** (POWER.md), not as "no connectivity".

## Deviations
Any change after this file is committed is listed here with a date and a reason.

- 2026-10-03: the first frozen settings (b554cdd) detect the planted graph in only 1 of 12 synthetic markets in rolling mode, because early stopping keeps pre-formation models and relu edges die. **Before any real-data run**, a third synthetic-only round fixes the optimization (validation-IC early stopping with a minimum epoch count, restarts of the first fit, and an optional signed message). Its selection rule is committed before its grid runs. The real run waits until the procedure reliably detects planted graphs. The success criteria above are unchanged.
