# M5: prior-anchored learned connectivity, power on synthetic intraday markets

No real data was used for anything in this file. The selection rule below was written and committed **before** the
tuning grid was run (see the git history of this file).

## What is tested

The question is whether learning a correction to MarketRank's chain beats the chain as it is.
- **learned:** `A_t = row-softmax over the prior support of (log P_prior + g(pair) + u_i^T v_j)`.
  - g is a small MLP over (log P, log raw, same sector, log size ratio); return correlation is omitted.
  - u and v have rank 4.
  - Optionally **signed**: the weights are multiplied by `tanh(b + g_sign + u'^T v')`.
- **Bprior:** A = P_prior, fixed (MarketRank's chain).
- **B0:** no message.
- **B0E:** a per-stock embedding with no message.

All variants share the encoder, the head, the message projection (8 dims) and the training protocol (`wf_m5.DEFAULTS`):
- one session (26 bars) per step, trained on the 19 labeled bars;
- expanding window, monthly retrain (21 sessions), first prediction at session 60, embargo of 1 session;
- validation-IC early stopping: patience 5, min epochs 6, max 30;
- 2 restarts per from-scratch fit;
- rolling fine-tune of at most 3 epochs, held out on the last 10 sessions.

Code: `intraday.py` (layout, label mask, features), `prior_model.py`, `wf_m5.py`, `synth_intraday.py`,
`power_m5.py`. Tests: `tests/test_m5.py`.

## Synthetic intraday market (`synth_intraday.py`)

**Shape:** N = 300 stocks (and one N = 1000 point), 250 sessions of 26 bars, 10 sectors.

**Prior:** a base graph P0 with about 20 kept out-edges per row, biased toward the same sector and larger names.
- It is re-estimated each session as a noisy P_s on the same support.
- It is written at each session's first bar.

**Planted truth:** `W_s = rownorm(P_s * exp(delta))`, where `delta = 1.0 * same-sector + 1.2 * a_i.b_j` (rank 2). On top of that:
- sign flips on about 15% of the support, following a low-rank pattern;
- one off-prior edge on 10% of the rows.

**How the truth moves prices:** the idiosyncratic shock of stock j at bar t' moves stock i over bars t'+1..t'+6 in total by `beta * W_s[i, j] * e_j`. This effect stays inside the session.

**Label:** `label_6[t]` is the sum of ret1 over bars t+1..t+6. It is defined only when bars t+1..t+7 are in the session of bar t.

**Oracle IC:** the mean per-bar Spearman of the true predictable part against `label_6`. beta is calibrated by bisection to hit the target oracle IC. Each row of the tables also reports:
- the prior-graph IC (what the observed prior predicts);
- the correction IC (the part of the signal that only a correction can capture).

**Nulls:**
- **null_a:** W = 0, so nothing is predictive.
- **null_b:** W = P_s at oracle IC 0.05, so only the prior is predictive. Bprior is the right model, and learned must not beat it.

## Pre-registered selection rule (fixed before the grid ran)

**Candidates:** `unsigned` (signed = false) and `signed` (signed = true). Everything else is the shared protocol.

**Tuning grid:** market seeds {201, 202, 203}, which are distinct from the evaluation seeds {1, 2, 3}.
- **Planted:** oracle IC {0.02, 0.03, 0.05} × {rolling, scratch} × model seeds {1, 2}. That gives 12 planted markets per IC level.
- **Nulls:** null_a and null_b × {rolling, scratch} × model seed 1. That gives 12 nulls.

**Detected:** learned − Bprior has t > 2 **and** learned − B0E has t > 2. Both are paired per-bar IC differences, with day-clustered SE: each session's mean difference is one unit.

**Pass:** detected in **at least 10 of the 12** planted markets at oracle IC 0.05, **and every null clean**:
- null_a: |t| < 2 for both learned − Bprior and learned − B0E;
- null_b: |t| < 2 for learned − Bprior. Here learned − B0E > 0 is expected, so it is only reported.

**Choice among passing configurations:**
1. the most detections at oracle IC 0.02, then at 0.03, then at 0.05;
2. if still tied, the one with fewer learned parameters.

**If nothing passes:** stop. Report it, and run no power stage and no real-data run.

**After selection, the frozen config goes through the power curve.** It uses evaluation market seeds {1, 2, 3}:
- oracle IC {0.02, 0.03, 0.05, 0.08} × {rolling, scratch} × model seeds {1, 2};
- null_a and null_b with both model seeds;
- one N = 1000 point at oracle IC 0.05, market seed 1, both modes.

**Disclosure.** Before the rule was written, one market with seed 999 (IC 0.05 and 0.08) was run once. It served only as a timing and sanity smoke test. Seed 999 is in neither grid.

## Results

(filled in after the grid ran; see `power_m5_results/`)
