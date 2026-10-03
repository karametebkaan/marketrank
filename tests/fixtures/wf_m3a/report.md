# Walk-forward run `fixture`

- params hash `16bc0909`: `core{pressure=dollar lift=off k_out=20 k_in=10 retention=0 dangling=teleport h_ref=uniform lambda=1 alpha=0.85 min_dv=1e+06 max_vr=0 vol_scale=0 hl_slow=1e+09 hl_fast=3 horizons=1;4;8;} rebalance=weekly warmup=100 elig_window=20 top_n=0 min_dv=0 ic_h=1;2;5;20; base=VOO:0.5;S0_01:0.29999999999999999;S0_02:0.20000000000000001; tilt=0.20000000000000001 k=5 max_name_tilt=0.10000000000000001 cost_bps=10 max_turnover=0.5 blend=12/1/12/4 gate_t=-1000000000`
- panel: 50 nodes, 400 bars (2025-01-02 to 2026-02-05)
- rebalances: 43 (weekly, 2025-04-13 to 2026-02-01), warm-up 100 bars
- registry: 9 trials, variance of daily Sharpe 0.000979; 9 IR trials, variance of daily IR 0.011981

## IC research table

Spearman IC of each signal's z-score (eligible names) against the open-to-open return over h bars, on non-overlapping samples. h = 20 is a decay diagnostic.

| signal | h | mean IC | t | n | positive years |
|---|---:|---:|---:|---:|---:|
| score | 1 | 0.0886 | 9.20 | 298 | 2 of 2 |
| score | 2 | 0.1206 | 9.06 | 149 | 2 of 2 |
| score | 5 | 0.1709 | 7.97 | 59 | 2 of 2 |
| score | 20 | 0.2614 | 6.08 | 14 | 1 of 1 |
| pulse1 | 1 | 0.0259 | 2.69 | 298 | 2 of 2 |
| pulse1 | 2 | 0.0274 | 2.21 | 149 | 1 of 2 |
| pulse1 | 5 | 0.0351 | 1.48 | 59 | 2 of 2 |
| pulse1 | 20 | 0.0126 | 0.28 | 14 | 1 of 1 |
| pulse5 | 1 | 0.0370 | 3.65 | 298 | 2 of 2 |
| pulse5 | 2 | 0.0453 | 3.08 | 149 | 2 of 2 |
| pulse5 | 5 | 0.0678 | 2.91 | 59 | 2 of 2 |
| pulse5 | 20 | 0.1119 | 2.17 | 14 | 1 of 1 |
| pulse20 | 1 | 0.0698 | 6.92 | 298 | 2 of 2 |
| pulse20 | 2 | 0.0994 | 6.98 | 149 | 2 of 2 |
| pulse20 | 5 | 0.1299 | 5.58 | 59 | 2 of 2 |
| pulse20 | 20 | 0.2219 | 4.35 | 14 | 1 of 1 |
| pi_rel_size | 1 | -0.0552 | -4.85 | 298 | 0 of 2 |
| pi_rel_size | 2 | -0.0770 | -4.76 | 149 | 0 of 2 |
| pi_rel_size | 5 | -0.0982 | -3.74 | 59 | 0 of 2 |
| pi_rel_size | 20 | -0.1369 | -2.39 | 14 | 0 of 1 |
| neg_hotness | 1 | -0.0886 | -9.20 | 298 | 0 of 2 |
| neg_hotness | 2 | -0.1206 | -9.06 | 149 | 0 of 2 |
| neg_hotness | 5 | -0.1709 | -7.97 | 59 | 0 of 2 |
| neg_hotness | 20 | -0.2614 | -6.08 | 14 | 0 of 1 |
| inflow_mom5 | 1 | 0.0593 | 4.89 | 298 | 2 of 2 |
| inflow_mom5 | 2 | 0.0753 | 4.12 | 149 | 2 of 2 |
| inflow_mom5 | 5 | 0.1172 | 4.27 | 59 | 2 of 2 |
| inflow_mom5 | 20 | 0.1504 | 2.83 | 14 | 1 of 1 |
| forecast | 1 | 0.0444 | 4.67 | 298 | 2 of 2 |
| forecast | 2 | 0.0571 | 4.01 | 149 | 2 of 2 |
| forecast | 5 | 0.0885 | 3.94 | 59 | 2 of 2 |
| forecast | 20 | 0.1874 | 3.30 | 14 | 1 of 1 |

## Strategies against buy-and-hold of the base

Costs 10.0 bps per side, tilt 0.20 over the top 5. Ann. excess is the mean daily excess return times 252 (not the gap between annualized returns). DSR_excess deflates the daily IR of the excess over buy-and-hold across all 9 IR trials in the registry (gate c3). DSR(total) deflates the total-return Sharpe across all 9 trials; it is informational only, since a high-Sharpe base passes it without any excess.

| curve | days | cum | ann | vol | Sharpe | max DD | ann excess | excess CI95 | IR | years + | turnover | costs | DSR_excess | DSR(total), informational |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| blend | 298 | -36.95% | -32.30% | 11.04% | -3.47 | 40.02% | 19.85% | 11.12% .. 28.99% | 6.63 | 1 of 1 | 4.69 | 0.810% | 1.000 | 0.000 |
| sig:score | 298 | -35.20% | -30.71% | 10.36% | -3.49 | 37.73% | 22.09% | 13.62% .. 30.38% | 6.89 | 1 of 1 | 0.72 | 0.214% | 1.000 | 0.000 |
| sig:pulse1 | 298 | -37.42% | -32.72% | 10.62% | -3.68 | 40.10% | 19.17% | 11.92% .. 26.66% | 6.15 | 1 of 1 | 7.38 | 1.313% | 1.000 | 0.000 |
| sig:pulse5 | 298 | -37.09% | -32.43% | 10.43% | -3.70 | 39.95% | 19.59% | 11.89% .. 27.42% | 5.94 | 1 of 1 | 7.32 | 1.315% | 1.000 | 0.000 |
| sig:pulse20 | 298 | -36.27% | -31.68% | 10.41% | -3.60 | 39.31% | 20.69% | 12.88% .. 28.10% | 6.26 | 1 of 1 | 4.33 | 0.825% | 1.000 | 0.000 |
| sig:pi_rel_size | 298 | -44.97% | -39.66% | 10.89% | -4.58 | 47.24% | 8.34% | 3.76% .. 13.06% | 3.06 | 1 of 1 | 1.80 | 0.384% | 0.675 | 0.000 |
| sig:neg_hotness | 298 | -47.42% | -41.94% | 11.09% | -4.84 | 49.49% | 4.52% | -0.19% .. 9.03% | 1.84 | 1 of 1 | 0.62 | 0.194% | 0.192 | 0.000 |
| sig:inflow_mom5 | 298 | -37.28% | -32.60% | 10.57% | -3.68 | 39.73% | 19.35% | 10.85% .. 28.57% | 5.88 | 1 of 1 | 7.68 | 1.361% | 1.000 | 0.000 |
| sig:forecast | 298 | -36.81% | -32.17% | 10.53% | -3.63 | 39.35% | 19.98% | 12.24% .. 27.63% | 6.09 | 1 of 1 | 6.26 | 1.137% | 1.000 | 0.000 |
| bench:buyhold | 298 | -50.30% | -44.63% | 12.95% | -4.49 | 52.90% | 0.00% | 0.00% .. 0.00% | n/a | 0 of 1 | 0.00 | 0.100% | - | - |
| bench:rebalanced | 298 | -50.35% | -44.69% | 12.86% | -4.53 | 53.01% | -0.11% | -0.72% .. 0.52% | -0.20 | 0 of 1 | 0.24 | 0.136% | - | - |
| bench:VOO | 298 | -45.33% | -39.99% | 15.83% | -3.14 | 47.20% | 8.46% | -2.19% .. 18.14% | 1.23 | 1 of 1 | 0.00 | 0.100% | - | - |
| bench:ew_eligible | 298 | 7.18% | 6.04% | 4.64% | 1.29 | 6.15% | 64.19% | 39.27% .. 87.85% | 5.38 | 1 of 1 | 0.49 | 0.198% | - | - |

## Secondary (reported, not gated)

Pre-registered on 2026-10-02 as secondaries for the next experiment (spec amendment). DSR_excess here uses the same registry IR trials as c3, an approximation since the benchmark differs.

### Strategies against the base rebalanced on the same calendar

This removes the rebalancing drag of the hindsight-selected base from the comparison.

| curve | days | ann | max DD | ann excess | excess CI95 | IR | years + | turnover | DSR_excess |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| blend | 298 | -32.30% | 40.02% | 19.95% | 11.02% .. 29.05% | 6.95 | 1 of 1 | 4.69 | 1.000 |
| sig:score | 298 | -30.71% | 37.73% | 22.20% | 13.80% .. 30.37% | 7.18 | 1 of 1 | 0.72 | 1.000 |
| sig:pulse1 | 298 | -32.72% | 40.10% | 19.28% | 12.07% .. 26.57% | 6.38 | 1 of 1 | 7.38 | 1.000 |
| sig:pulse5 | 298 | -32.43% | 39.95% | 19.70% | 12.16% .. 27.53% | 6.18 | 1 of 1 | 7.32 | 1.000 |
| sig:pulse20 | 298 | -31.68% | 39.31% | 20.80% | 13.25% .. 28.09% | 6.54 | 1 of 1 | 4.33 | 1.000 |
| sig:pi_rel_size | 298 | -39.66% | 47.24% | 8.45% | 3.80% .. 13.23% | 3.21 | 1 of 1 | 1.80 | 0.731 |
| sig:neg_hotness | 298 | -41.94% | 49.49% | 4.63% | -0.21% .. 9.11% | 1.98 | 1 of 1 | 0.62 | 0.237 |
| sig:inflow_mom5 | 298 | -32.60% | 39.73% | 19.46% | 11.06% .. 28.47% | 6.13 | 1 of 1 | 7.68 | 1.000 |
| sig:forecast | 298 | -32.17% | 39.35% | 20.09% | 12.32% .. 27.60% | 6.36 | 1 of 1 | 6.26 | 1.000 |

### Base-free sleeves against the equal-weight eligible universe

Each sleeve holds the top 5 names at 1/5 each (tilt 1, no base; a flat blend holds the benchmark), with the same costs and turnover cap, against equal weight over each rebalance's eligible names rebalanced on the same calendar (`bench:ew_eligible`). Free of the hindsight-selected base.

| curve | days | ann | max DD | ann excess | excess CI95 | IR | years + | turnover | DSR_excess |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| sleeve:blend | 298 | 60.69% | 2.28% | 41.83% | 20.67% .. 64.60% | 6.44 | 1 of 1 | 15.50 | 1.000 |
| sleeve:score | 298 | 66.27% | 2.96% | 45.24% | 27.51% .. 62.54% | 6.98 | 1 of 1 | 2.20 | 1.000 |
| sleeve:pulse1 | 298 | 51.91% | 2.18% | 36.16% | 20.98% .. 52.65% | 6.08 | 1 of 1 | 21.00 | 1.000 |
| sleeve:pulse5 | 298 | 50.63% | 2.45% | 35.32% | 19.68% .. 51.73% | 5.65 | 1 of 1 | 21.00 | 1.000 |
| sleeve:pulse20 | 298 | 56.69% | 3.11% | 39.34% | 23.14% .. 56.94% | 5.40 | 1 of 1 | 17.99 | 0.999 |
| sleeve:pi_rel_size | 298 | -15.34% | 18.94% | -22.27% | -38.03% .. -8.07% | -3.13 | 0 of 1 | 7.90 | 0.000 |
| sleeve:neg_hotness | 298 | -29.92% | 34.33% | -41.19% | -57.28% .. -26.62% | -5.99 | 0 of 1 | 2.02 | 0.000 |
| sleeve:inflow_mom5 | 298 | 53.98% | 2.08% | 37.52% | 20.30% .. 57.63% | 6.02 | 1 of 1 | 21.00 | 1.000 |
| sleeve:forecast | 298 | 58.39% | 3.05% | 40.40% | 23.33% .. 58.02% | 5.83 | 1 of 1 | 20.80 | 1.000 |

`bench:ew_eligible` itself: ann 6.04%, max DD 6.15%, turnover 0.49, costs 0.198%.

## Blend

Gate open in 31 of 43 periods (72.1%); weights formed in 36 periods. Mean weight over those periods:

| signal | mean w |
|---|---:|
| score | 0.217 |
| pulse1 | 0.072 |
| pulse5 | 0.124 |
| pulse20 | 0.195 |
| pi_rel_size | 0.011 |
| neg_hotness | 0.002 |
| inflow_mom5 | 0.215 |
| forecast | 0.165 |

The full weight history is in results.json (`blend`).

## Decision gate (blend)

| criterion | result |
|---|---|
| c1: annual excess > 0 and CI95 lower bound > 0 | pass |
| c2: excess positive in >= 60% of years | pass |
| c3: deflated IR (excess vs buy-and-hold) > 0.95 | pass (DSR_excess 1.000; DSR(total) 0.000, informational) |
| c4: max drawdown <= base 52.90% + 5% | pass |
| c5: holds on the large-cap sub-universe | pending (run again with --wf-top-n 500 and pass --wf-largecap-run) |

**Verdict: pending** (c1 to c4 pass; c5 not yet run).

Advisory only. The universe is today's snapshot, so delisted names are missing (survivorship bias).
