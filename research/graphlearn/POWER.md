# Learned graph: power curve and frozen settings (synthetic markets only)

No real data was used to choose anything in this file.

## Pre-registered selection rule (written and committed before the tuning grid was run)

**Tuning markets.** The synthetic planted market (`synth.py`) with N=300 and T=1500 bars, which gives 300 weekly rebalances and 143 predicted dates.
- beta ∈ {0.2, 0.3, 0.4} on seeds {101, 102}: six points;
- the null market (W=0) on seeds {101, 102}.

The tuning seeds are disjoint from the seeds the power curve reports ({1, 2}).
Every point runs `train_wf.py` exactly as deployed:
- rolling mode, retrain every 13, embargo 1;
- variants learned, B0 and B0E.

**Candidates.** The starting setting:

| rank | topk | emb-lr | emb-l2 | l1 | finetune-epochs | ft-holdout | window | patience | epochs |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 20 | 1e-3 | 1e-3 | 1e-2 | 2 | 13 | 156 | 10 | 50 |

Each of the following one-at-a-time variations of it is also a candidate:
- emb-lr 1e-2;
- emb-l2 1e-4 or 1e-2;
- rank 4 or 16;
- l1 1e-3 or 1e-1;
- finetune-epochs 5.

**Rule.**
1. **Admissible:** on both null markets, |t| < 2 for learned − B0 and for learned − B0E.
2. **Score:** the number of the six tuning points where learned − B0 has t > 2 **and** learned − B0E has t > 2. This is detection power against both controls.
3. **Choice:** the admissible setting with the highest score. Ties go to the higher mean, over the six points, of min(t_B0, t_B0E).
4. **One refinement round:** if two or more variations each beat the start under rule 3, their combination is evaluated on the same markets. It replaces the winner only if it is admissible and better under rule 3.
5. **Fallback:** if nothing is admissible, the starting setting is frozen, and this is reported as a failure of the null.

The frozen setting then goes through the power curve:
- beta ∈ {0.1, 0.2, 0.3, 0.4, 0.6, 0.8} × seeds {1, 2} × {rolling, scratch};
- the null market;
- one N=1000 point with 30% eligibility churn (beta 0.4).

**"Smallest oracle IC detectable at t > 2"** is the smallest oracle IC at which both reported seeds, in rolling mode, have learned − B0 t > 2 **and** learned − B0E t > 2.

## Amendment before round 2 (written and committed after round 1, before running round 2)

**Round 1 outcome under the rule.**
- Every candidate was admissible.
- The winner was **emb-lr 1e-2**: score 1 of 6, mean min-t 0.96.
- Three variations beat the start under rule 3: emb-lr 1e-2, finetune-epochs 5 and l1 1e-1. Rule step 4 therefore evaluates their combination, `combo`.

**Diagnosis that motivates round 2: the beta=0.4 AUC of 0.43.**
- I reproduced it with the first-round code on the same market, with the old calendar (no +1 bar) and 4 threads: AUC 0.425 and IC difference t = 0.7.
- The first scratch fit early-stopped at its best epoch 0. At that point the head read the neighbours with a slightly *negative* sign: the Jacobian ∂ŷ_i/∂r5_j over the top-k edges was about −1e-4.
- The rolling fine-tunes then amplified that sign. The Jacobian over false edges went to about −0.015 and over true edges negative too.
- With a negative read-out, the straight-through gradient pushes **true** edges down, because they predict with the "wrong" sign. They hit relu = 0 and die: 52% of the true edges had S = 0 at the end, against 0% at the start. S_true fell from 1.15 to 0.75, while the S of other pairs rose.
- relu makes this absorbing: a dead edge gets no gradient. So AUC drifts systematically below 0.5.
- The same market with a different trajectory recovered: with 13 epochs and best epoch 2, AUC was 0.82 and t = 10, because the read-out sign came out positive.
- Round 1 shows the same signature: on seed 101 most candidates end at AUC 0.42–0.47.

**Round 2 candidates**, on the same tuning markets and in the same rolling mode:
- `combo`: emb-lr 1e-2, finetune-epochs 5, l1 1e-1 (rule step 4);
- `emb_lr=0.01+softplus` and `combo+softplus`: S = softplus(E_s E_dᵀ) instead of relu, so a suppressed edge is never dead and can come back if the read-out sign flips;
- `prev` and `prev+softplus`: the first-round defaults (rank 16, emb-lr 1e-2, emb-l2 0, l1 1e-4, finetune-epochs 5), now run with the honest fine-tune holdout.

**Rule:** unchanged (rules 1–3), applied to every round-1 and round-2 candidate together. The winner is frozen; there is no further round.


## Results

All runs below were on synthetic markets. All raw rows are in `power_results/`: `tune.jsonl`, `tune2.jsonl`, `power.jsonl`, `sens.jsonl` and `selection_tune2.json`. Reproduce them with `power_curve.py --out DIR --stage tune|tune2|power|sens`.

**Column definitions.**
- **oracle IC:** the mean per-date Spearman of the *true* predictable signal against the label. It is the ceiling for any model.
- **learned − B0 / learned − B0E:** the paired per-date IC difference, with its t-statistic.
- **edge AUC:** the final retrain's learned S, true edges against false edges, over the rows that have true edges.
- **1st best epoch:** the epoch that the first, from-scratch fit restored.

### Selection (rounds 1 and 2, all candidates, tuning seeds 101/102, rolling)

| candidate | admissible (null \|t\|<2) | score (of 6) | tie-break: mean min(t_B0, t_B0E) |
|---|---|---|---|
| combo+softplus | yes | 2 | +0.43 |
| emb_lr=0.01+softplus | yes | 1 | +1.18 |
| emb_lr=0.01 | yes | 1 | +0.96 |
| finetune_epochs=5 | yes | 0 | +0.61 |
| l1=0.1 | yes | 0 | +0.31 |
| combo | yes | 0 | +0.26 |
| start | yes | 0 | -0.20 |
| emb_l2=0.01 | yes | 0 | -0.26 |
| emb_l2=0.0001 | yes | 0 | -0.28 |
| l1=0.001 | yes | 0 | -0.29 |
| rank=4 | yes | 0 | -0.42 |
| rank=16 | yes | 0 | -0.54 |
| prev+softplus | no | 3 | +2.03 |
| prev | no | 2 | +1.62 |

**Frozen setting: `combo+softplus`.** The rule ranks it highest among the admissible candidates.
- Values: rank 8, topk 20, score softplus, emb-lr 1e-2, emb-l2 1e-3, l1 1e-1, finetune-epochs 5, ft-holdout 13, window 156, patience 10, epochs 50, embargo 1, retrain every 13.
- Stored in `frozen_settings.json`; these are the CLI defaults.

`prev+softplus` had the most tuning power (score 3). It was inadmissible because on the null market with seed 102, learned − B0E had t = −2.4: learned was significantly *worse* than the control.

### Power curve of the frozen setting (reported seeds 1/2)

| setting | N | beta | seed | null/churn | mode | oracle IC | learned IC | learned-B0 (t) | learned-B0E (t) | B0E-B0 (t) | edge AUC | 1st best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | 300 | 0.1 | 1 |  | rolling | 0.054 | +0.004 | -0.002 (-0.2) | -0.001 (-0.1) | -0.001 (-0.1) | 0.551 | 3 |
| frozen | 300 | 0.2 | 1 |  | rolling | 0.114 | +0.009 | +0.003 (+0.5) | +0.004 (+0.7) | -0.001 (-0.2) | 0.552 | 2 |
| frozen | 300 | 0.3 | 1 |  | rolling | 0.175 | -0.001 | -0.007 (-1.0) | -0.004 (-0.6) | -0.003 (-0.4) | 0.544 | 3 |
| frozen | 300 | 0.4 | 1 |  | rolling | 0.236 | +0.004 | -0.001 (-0.2) | +0.003 (+0.4) | -0.004 (-0.6) | 0.536 | 3 |
| frozen | 300 | 0.6 | 1 |  | rolling | 0.364 | +0.008 | -0.000 (-0.1) | +0.003 (+0.4) | -0.003 (-0.5) | 0.522 | 3 |
| frozen | 300 | 0.8 | 1 |  | rolling | 0.503 | +0.001 | -0.009 (-1.1) | -0.005 (-0.6) | -0.004 (-0.5) | 0.522 | 2 |
| frozen | 300 | 0.8 | 1 | null | rolling | 0.000 | -0.003 | -0.004 (-0.7) | -0.006 (-0.9) | +0.002 (+0.3) | nan | 3 |
| frozen | 300 | 0.1 | 2 |  | rolling | 0.058 | +0.004 | +0.003 (+0.5) | +0.004 (+0.6) | -0.001 (-0.1) | 0.541 | 2 |
| frozen | 300 | 0.2 | 2 |  | rolling | 0.118 | +0.000 | +0.001 (+0.1) | +0.002 (+0.3) | -0.001 (-0.2) | 0.552 | 2 |
| frozen | 300 | 0.3 | 2 |  | rolling | 0.178 | +0.003 | +0.004 (+0.7) | +0.005 (+0.7) | -0.001 (-0.1) | 0.540 | 2 |
| frozen | 300 | 0.4 | 2 |  | rolling | 0.239 | -0.006 | -0.002 (-0.3) | -0.003 (-0.5) | +0.001 (+0.1) | 0.541 | 2 |
| frozen | 300 | 0.6 | 2 |  | rolling | 0.367 | -0.005 | +0.003 (+0.4) | -0.004 (-0.6) | +0.007 (+1.0) | 0.552 | 2 |
| frozen | 300 | 0.8 | 2 |  | rolling | 0.503 | +0.157 | +0.165 (+14.9) | +0.158 (+13.2) | +0.007 (+0.9) | 0.711 | 0 |
| frozen | 300 | 0.8 | 2 | null | rolling | 0.000 | -0.001 | -0.007 (-1.0) | +0.001 (+0.2) | -0.008 (-1.3) | nan | 2 |
| frozen | 1000 | 0.4 | 1 | churn 0.3 | rolling | 0.246 | +0.001 | -0.003 (-0.7) | -0.002 (-0.5) | -0.001 (-0.2) | 0.492 | 2 |
| frozen | 300 | 0.1 | 1 |  | scratch | 0.054 | +0.009 | +0.014 (+2.4) | +0.007 (+1.1) | +0.007 (+1.1) | 0.559 | 3 |
| frozen | 300 | 0.2 | 1 |  | scratch | 0.114 | +0.006 | +0.010 (+1.8) | +0.005 (+0.7) | +0.006 (+0.9) | 0.557 | 2 |
| frozen | 300 | 0.3 | 1 |  | scratch | 0.175 | +0.010 | +0.015 (+2.5) | +0.012 (+1.7) | +0.004 (+0.6) | 0.559 | 3 |
| frozen | 300 | 0.4 | 1 |  | scratch | 0.236 | +0.014 | +0.016 (+2.5) | +0.014 (+2.1) | +0.002 (+0.3) | 0.552 | 3 |
| frozen | 300 | 0.6 | 1 |  | scratch | 0.364 | +0.031 | +0.033 (+3.8) | +0.031 (+3.5) | +0.002 (+0.3) | 0.547 | 3 |
| frozen | 300 | 0.8 | 1 |  | scratch | 0.503 | +0.064 | +0.061 (+5.3) | +0.062 (+5.0) | -0.000 (-0.0) | 0.550 | 2 |
| frozen | 300 | 0.8 | 1 | null | scratch | 0.000 | +0.013 | +0.019 (+2.9) | +0.008 (+1.2) | +0.011 (+1.9) | nan | 3 |
| frozen | 300 | 0.1 | 2 |  | scratch | 0.058 | -0.001 | -0.005 (-0.8) | -0.002 (-0.3) | -0.003 (-0.4) | 0.536 | 2 |
| frozen | 300 | 0.2 | 2 |  | scratch | 0.118 | -0.002 | -0.004 (-0.7) | -0.001 (-0.2) | -0.003 (-0.5) | 0.530 | 2 |
| frozen | 300 | 0.3 | 2 |  | scratch | 0.178 | -0.001 | -0.000 (-0.1) | +0.000 (+0.0) | -0.001 (-0.1) | 0.529 | 2 |
| frozen | 300 | 0.4 | 2 |  | scratch | 0.239 | +0.001 | +0.005 (+0.9) | +0.003 (+0.4) | +0.002 (+0.4) | 0.523 | 2 |
| frozen | 300 | 0.6 | 2 |  | scratch | 0.367 | +0.030 | +0.036 (+4.8) | +0.034 (+4.4) | +0.002 (+0.4) | 0.538 | 2 |
| frozen | 300 | 0.8 | 2 |  | scratch | 0.503 | +0.125 | +0.128 (+10.4) | +0.127 (+9.5) | +0.001 (+0.2) | 0.538 | 0 |
| frozen | 300 | 0.8 | 2 | null | scratch | 0.000 | +0.004 | -0.001 (-0.1) | +0.000 (+0.0) | -0.001 (-0.2) | nan | 2 |
| frozen | 1000 | 0.4 | 1 | churn 0.3 | scratch | 0.246 | +0.001 | -0.005 (-1.2) | -0.002 (-0.4) | -0.003 (-1.2) | 0.502 | 2 |

### Sensitivity: the most powerful tuning candidate, `prev+softplus` (not frozen; rolling)

| setting | N | beta | seed | null/churn | mode | oracle IC | learned IC | learned-B0 (t) | learned-B0E (t) | B0E-B0 (t) | edge AUC | 1st best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prev+softplus | 300 | 0.1 | 1 |  | rolling | 0.054 | -0.010 | -0.016 (-2.4) | -0.011 (-1.8) | -0.005 (-0.8) | 0.551 | 0 |
| prev+softplus | 300 | 0.2 | 1 |  | rolling | 0.114 | +0.005 | -0.001 (-0.1) | +0.002 (+0.3) | -0.003 (-0.4) | 0.569 | 0 |
| prev+softplus | 300 | 0.3 | 1 |  | rolling | 0.175 | +0.016 | +0.010 (+1.3) | +0.012 (+1.8) | -0.002 (-0.3) | 0.658 | 1 |
| prev+softplus | 300 | 0.4 | 1 |  | rolling | 0.236 | -0.002 | -0.008 (-1.2) | -0.004 (-0.6) | -0.003 (-0.5) | 0.562 | 1 |
| prev+softplus | 300 | 0.6 | 1 |  | rolling | 0.364 | +0.189 | +0.181 (+20.7) | +0.188 (+21.0) | -0.008 (-1.1) | 0.884 | 8 |
| prev+softplus | 300 | 0.8 | 1 |  | rolling | 0.503 | +0.372 | +0.362 (+36.7) | +0.368 (+37.1) | -0.006 (-0.7) | 0.975 | 17 |
| prev+softplus | 300 | 0.8 | 1 | null | rolling | 0.000 | +0.001 | +0.000 (+0.1) | +0.003 (+0.4) | -0.002 (-0.4) | nan | 0 |
| prev+softplus | 300 | 0.1 | 2 |  | rolling | 0.058 | -0.004 | -0.005 (-0.7) | -0.000 (-0.1) | -0.005 (-0.7) | 0.496 | 0 |
| prev+softplus | 300 | 0.2 | 2 |  | rolling | 0.118 | -0.008 | -0.008 (-1.2) | -0.003 (-0.5) | -0.005 (-0.7) | 0.475 | 0 |
| prev+softplus | 300 | 0.3 | 2 |  | rolling | 0.178 | -0.006 | -0.005 (-0.7) | -0.001 (-0.1) | -0.004 (-0.6) | 0.433 | 0 |
| prev+softplus | 300 | 0.4 | 2 |  | rolling | 0.239 | -0.007 | -0.003 (-0.5) | -0.001 (-0.2) | -0.002 (-0.3) | 0.446 | 0 |
| prev+softplus | 300 | 0.6 | 2 |  | rolling | 0.367 | -0.002 | +0.006 (+0.9) | +0.003 (+0.4) | +0.003 (+0.4) | 0.446 | 0 |
| prev+softplus | 300 | 0.8 | 2 |  | rolling | 0.503 | -0.002 | +0.006 (+0.9) | -0.001 (-0.2) | +0.008 (+1.0) | 0.425 | 0 |
| prev+softplus | 300 | 0.8 | 2 | null | rolling | 0.000 | +0.004 | -0.003 (-0.5) | +0.008 (+1.3) | -0.011 (-1.8) | nan | 0 |

### Smallest oracle IC detectable at t > 2

The criterion: both reported seeds, learned − B0 **and** learned − B0E, t > 2.
- **Frozen setting, rolling (the deployed regime): none up to an oracle IC of 0.50 (beta 0.8).** Only seed 2 at beta 0.8 is detected (t = 15). Seed 1 is not detected even there.
- Frozen setting, scratch mode: an oracle IC of **0.36** (beta 0.6; t = 3.5 to 4.8). At beta ≤ 0.4, only seed 1 is detected, at t ≈ 2.1 to 2.5. But scratch mode's null market with seed 1 has learned − B0 at t = +2.9, so scratch mode does not keep the null clean against B0. (It does against B0E: t = 1.2.)
- `prev+softplus`, rolling: seed 1 is detected at an oracle IC of 0.36 and 0.50 (t = 21 and 37; AUC 0.88 and 0.975). Seed 2 is never detected (AUC 0.43 to 0.50). So it does not meet the both-seeds criterion either.
- N=1000 with 30% churn, beta 0.4 (oracle 0.25): not detected in either mode.

For scale, a realistic weekly cross-sectional IC from any source is 0.01 to 0.05. The best case above needs an oracle IC 7 to 30 times larger.

### What decides success: the first fit's early-stopping epoch

Across every table, a run either finds the graph clearly (AUC 0.7 to 0.98, t > 10) or not at all (AUC ≈ 0.43 to 0.55). Which one happens tracks the epoch the first from-scratch fit restored:
- the found cases restored epoch 8 or 17 (`prev+softplus`, seed 1, beta 0.6 and 0.8);
- the failures restored epoch 0 to 3. The exception is the one frozen hit, which restored epoch 0.

**Mechanism** (instrumented in the round-2 amendment):
1. The graph forms only after a plateau of several epochs, during which the validation loss does not improve.
2. Early stopping with patience 10 therefore often restores a pre-graph model.
3. If that model's read-out of the neighbours has a negative sign, the gradient pushes the true edges down, and the run locks into "no graph". Under relu the true edges die (52% at S = 0); softplus keeps them alive but does not reverse the sign.
4. In rolling mode, the honest fine-tunes then early-stop on a 13-date holdout. They mostly keep the pre-fine-tune model (best epoch −1), so a failed start is never repaired.

### Conclusion

- **Detection power.** With the frozen, pre-registered settings, the deployed rolling regime has essentially no power on these synthetic markets: only 1 of 12 planted points is detected, while the null stays clean.
- **What it means for a real run.** A real-data run would be uninformative in its negative direction. Without a change of procedure it would not tell us whether the connectivity can be learned.
- **What the failure looks like.** It is an optimization pathology: a bimodal outcome decided by the first fit's stopping epoch and the read-out sign. It is not a statistical limit. When the graph is found, it is found clearly: AUC 0.975 and IC +0.36.

Candidate fixes, untested here and not part of the frozen setting; they need a new pre-registered round:
- a minimum number of epochs before early stopping, or early stopping on validation IC;
- several restarts per first fit, chosen by validation;
- a signed or two-sided message (separate positive and negative adjacency heads), so the read-out sign cannot lock the edges.

Until then, any real-data comparison should be read as "not detectable with this procedure", not as "no connectivity".

## Appendix: every tuning point (seeds 101/102, rolling)

| setting | N | beta | seed | null/churn | mode | oracle IC | learned IC | learned-B0 (t) | learned-B0E (t) | B0E-B0 (t) | edge AUC | 1st best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.005 | -0.001 (-0.1) | +0.012 (+2.0) | -0.012 (-2.2) | 0.451 | 2 |
| start | 300 | 0.3 | 101 |  | rolling | 0.177 | -0.003 | +0.001 (+0.1) | +0.012 (+2.0) | -0.011 (-2.1) | 0.473 | 6 |
| start | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.002 | +0.006 (+1.1) | +0.017 (+3.0) | -0.011 (-2.5) | 0.461 | 3 |
| start | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.007 | -0.002 (-0.4) | +0.011 (+1.9) | -0.013 (-2.4) | nan | 4 |
| start | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.001 | -0.004 (-0.6) | -0.006 (-1.0) | +0.003 (+0.4) | 0.535 | 1 |
| start | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.003 | +0.002 (+0.4) | -0.006 (-1.0) | +0.008 (+1.5) | 0.521 | 1 |
| start | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.006 | +0.005 (+0.8) | -0.002 (-0.4) | +0.007 (+1.1) | 0.534 | 1 |
| start | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.005 | -0.002 (-0.3) | -0.002 (-0.3) | +0.000 (+0.1) | nan | 0 |
| emb_lr=0.01 | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.000 | +0.004 (+0.8) | +0.008 (+1.4) | -0.004 (-0.6) | 0.524 | 0 |
| emb_lr=0.01 | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.005 | +0.009 (+1.6) | +0.013 (+2.0) | -0.004 (-0.6) | 0.493 | 0 |
| emb_lr=0.01 | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.014 | +0.019 (+3.1) | +0.022 (+3.4) | -0.003 (-0.6) | 0.562 | 2 |
| emb_lr=0.01 | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.003 | +0.001 (+0.2) | +0.004 (+0.7) | -0.003 (-0.5) | nan | 0 |
| emb_lr=0.01 | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.005 | -0.000 (-0.0) | -0.001 (-0.2) | +0.001 (+0.2) | 0.536 | 0 |
| emb_lr=0.01 | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.004 | +0.004 (+0.6) | -0.004 (-0.6) | +0.008 (+1.3) | 0.571 | 0 |
| emb_lr=0.01 | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.015 | +0.014 (+2.2) | +0.006 (+1.0) | +0.007 (+1.3) | 0.596 | 0 |
| emb_lr=0.01 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.003 | -0.003 (-0.6) | -0.002 (-0.3) | -0.001 (-0.2) | nan | 0 |
| emb_l2=0.0001 | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.007 | -0.002 (-0.4) | +0.010 (+1.7) | -0.012 (-2.2) | 0.457 | 4 |
| emb_l2=0.0001 | 300 | 0.3 | 101 |  | rolling | 0.177 | -0.004 | +0.000 (+0.1) | +0.012 (+2.0) | -0.011 (-2.1) | 0.464 | 4 |
| emb_l2=0.0001 | 300 | 0.4 | 101 |  | rolling | 0.238 | -0.000 | +0.004 (+0.8) | +0.015 (+2.6) | -0.011 (-2.5) | 0.458 | 2 |
| emb_l2=0.0001 | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.008 | -0.004 (-0.7) | +0.009 (+1.5) | -0.013 (-2.4) | nan | 4 |
| emb_l2=0.0001 | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.003 | -0.002 (-0.3) | -0.004 (-0.6) | +0.003 (+0.4) | 0.530 | 1 |
| emb_l2=0.0001 | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.005 | +0.004 (+0.8) | -0.004 (-0.7) | +0.008 (+1.5) | 0.528 | 1 |
| emb_l2=0.0001 | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.004 | +0.002 (+0.3) | -0.005 (-0.7) | +0.007 (+1.1) | 0.532 | 1 |
| emb_l2=0.0001 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.008 | +0.002 (+0.3) | +0.001 (+0.2) | +0.000 (+0.1) | nan | 0 |
| emb_l2=0.01 | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.006 | -0.001 (-0.2) | +0.011 (+1.9) | -0.012 (-2.2) | 0.452 | 2 |
| emb_l2=0.01 | 300 | 0.3 | 101 |  | rolling | 0.177 | -0.002 | +0.002 (+0.4) | +0.014 (+2.3) | -0.011 (-2.1) | 0.473 | 6 |
| emb_l2=0.01 | 300 | 0.4 | 101 |  | rolling | 0.238 | -0.001 | +0.004 (+0.7) | +0.015 (+2.7) | -0.011 (-2.5) | 0.465 | 4 |
| emb_l2=0.01 | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.010 | -0.005 (-0.9) | +0.008 (+1.3) | -0.013 (-2.4) | nan | 4 |
| emb_l2=0.01 | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.003 | -0.002 (-0.3) | -0.004 (-0.7) | +0.003 (+0.4) | 0.529 | 1 |
| emb_l2=0.01 | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.003 | +0.003 (+0.5) | -0.006 (-0.9) | +0.009 (+1.5) | 0.524 | 1 |
| emb_l2=0.01 | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.003 | +0.001 (+0.2) | -0.005 (-0.9) | +0.007 (+1.1) | 0.520 | 1 |
| emb_l2=0.01 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.005 | -0.002 (-0.4) | -0.002 (-0.4) | +0.000 (+0.1) | nan | 0 |
| rank=4 | 300 | 0.2 | 101 |  | rolling | 0.117 | +0.003 | +0.008 (+1.5) | +0.002 (+0.4) | +0.005 (+1.0) | 0.551 | 2 |
| rank=4 | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.001 | +0.005 (+1.0) | -0.001 (-0.2) | +0.006 (+1.2) | 0.549 | 2 |
| rank=4 | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.005 | +0.009 (+2.1) | +0.004 (+0.7) | +0.006 (+1.3) | 0.548 | 2 |
| rank=4 | 300 | 0.8 | 101 | null | rolling | 0.000 | +0.005 | +0.009 (+1.8) | +0.005 (+0.9) | +0.005 (+0.8) | nan | 2 |
| rank=4 | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.004 | -0.001 (-0.2) | -0.007 (-1.1) | +0.006 (+1.2) | 0.457 | 1 |
| rank=4 | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.004 | +0.003 (+0.6) | -0.006 (-0.9) | +0.009 (+1.6) | 0.455 | 2 |
| rank=4 | 300 | 0.4 | 102 |  | rolling | 0.243 | -0.000 | -0.002 (-0.3) | -0.009 (-1.4) | +0.007 (+1.3) | 0.451 | 2 |
| rank=4 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.007 | -0.000 (-0.0) | -0.004 (-0.7) | +0.004 (+0.7) | nan | 1 |
| rank=16 | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.005 | -0.000 (-0.0) | +0.002 (+0.3) | -0.002 (-0.4) | 0.420 | 2 |
| rank=16 | 300 | 0.3 | 101 |  | rolling | 0.177 | -0.003 | +0.001 (+0.3) | +0.008 (+1.2) | -0.006 (-1.1) | 0.413 | 2 |
| rank=16 | 300 | 0.4 | 101 |  | rolling | 0.238 | -0.002 | +0.003 (+0.4) | +0.008 (+1.3) | -0.006 (-1.0) | 0.494 | 2 |
| rank=16 | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.007 | -0.003 (-0.5) | +0.000 (+0.0) | -0.003 (-0.5) | nan | 4 |
| rank=16 | 300 | 0.2 | 102 |  | rolling | 0.122 | -0.001 | -0.006 (-1.0) | -0.007 (-1.0) | +0.001 (+0.1) | 0.495 | 0 |
| rank=16 | 300 | 0.3 | 102 |  | rolling | 0.182 | -0.006 | -0.006 (-0.9) | -0.011 (-1.6) | +0.005 (+0.9) | 0.510 | 0 |
| rank=16 | 300 | 0.4 | 102 |  | rolling | 0.243 | -0.000 | -0.002 (-0.3) | -0.010 (-1.3) | +0.008 (+1.2) | 0.526 | 0 |
| rank=16 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.002 | -0.005 (-0.7) | -0.006 (-0.8) | +0.001 (+0.2) | nan | 0 |
| l1=0.001 | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.008 | -0.003 (-0.5) | +0.009 (+1.6) | -0.012 (-2.2) | 0.458 | 2 |
| l1=0.001 | 300 | 0.3 | 101 |  | rolling | 0.177 | -0.003 | +0.002 (+0.3) | +0.013 (+2.1) | -0.011 (-2.1) | 0.463 | 2 |
| l1=0.001 | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.000 | +0.005 (+0.8) | +0.016 (+2.5) | -0.011 (-2.5) | 0.479 | 3 |
| l1=0.001 | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.008 | -0.003 (-0.5) | +0.010 (+1.7) | -0.013 (-2.4) | nan | 4 |
| l1=0.001 | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.003 | -0.002 (-0.4) | -0.004 (-0.7) | +0.003 (+0.4) | 0.536 | 0 |
| l1=0.001 | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.003 | +0.003 (+0.5) | -0.006 (-1.0) | +0.008 (+1.5) | 0.538 | 0 |
| l1=0.001 | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.005 | +0.003 (+0.5) | -0.004 (-0.6) | +0.007 (+1.1) | 0.554 | 0 |
| l1=0.001 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.004 | -0.003 (-0.5) | -0.003 (-0.5) | +0.000 (+0.1) | nan | 0 |
| l1=0.1 | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.009 | -0.004 (-0.8) | +0.008 (+1.3) | -0.012 (-2.2) | 0.447 | 6 |
| l1=0.1 | 300 | 0.3 | 101 |  | rolling | 0.177 | -0.004 | +0.000 (+0.1) | +0.012 (+1.9) | -0.011 (-2.1) | 0.456 | 6 |
| l1=0.1 | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.007 | +0.011 (+2.0) | +0.022 (+3.7) | -0.011 (-2.5) | 0.463 | 19 |
| l1=0.1 | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.010 | -0.006 (-1.1) | +0.007 (+1.1) | -0.013 (-2.4) | nan | 6 |
| l1=0.1 | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.010 | +0.005 (+0.8) | +0.003 (+0.4) | +0.003 (+0.4) | 0.486 | 2 |
| l1=0.1 | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.010 | +0.010 (+1.6) | +0.002 (+0.2) | +0.008 (+1.5) | 0.481 | 2 |
| l1=0.1 | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.008 | +0.006 (+1.0) | -0.000 (-0.0) | +0.007 (+1.1) | 0.499 | 2 |
| l1=0.1 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.012 | +0.005 (+0.8) | +0.004 (+0.6) | +0.000 (+0.1) | nan | 2 |
| finetune_epochs=5 | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.003 | +0.002 (+0.4) | +0.011 (+1.8) | -0.008 (-1.5) | 0.460 | 2 |
| finetune_epochs=5 | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.002 | +0.008 (+1.2) | +0.015 (+2.5) | -0.008 (-1.4) | 0.484 | 6 |
| finetune_epochs=5 | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.007 | +0.012 (+1.8) | +0.019 (+3.0) | -0.007 (-1.5) | 0.495 | 3 |
| finetune_epochs=5 | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.003 | +0.005 (+0.8) | +0.011 (+1.8) | -0.006 (-1.0) | nan | 4 |
| finetune_epochs=5 | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.004 | -0.000 (-0.0) | +0.001 (+0.1) | -0.001 (-0.1) | 0.536 | 1 |
| finetune_epochs=5 | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.007 | +0.006 (+1.1) | +0.001 (+0.2) | +0.005 (+0.8) | 0.529 | 1 |
| finetune_epochs=5 | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.007 | +0.008 (+1.3) | +0.000 (+0.1) | +0.008 (+1.3) | 0.557 | 1 |
| finetune_epochs=5 | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.004 | -0.001 (-0.2) | +0.001 (+0.1) | -0.002 (-0.3) | nan | 0 |
| combo | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.002 | +0.003 (+0.6) | +0.005 (+0.8) | -0.001 (-0.3) | 0.519 | 0 |
| combo | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.005 | +0.011 (+1.8) | +0.012 (+1.8) | -0.001 (-0.2) | 0.509 | 0 |
| combo | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.005 | +0.011 (+1.7) | +0.017 (+2.6) | -0.006 (-1.1) | 0.546 | 2 |
| combo | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.002 | +0.006 (+1.1) | +0.005 (+0.8) | +0.001 (+0.2) | nan | 0 |
| combo | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.004 | -0.000 (-0.1) | -0.004 (-0.5) | +0.003 (+0.6) | 0.515 | 0 |
| combo | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.002 | +0.000 (+0.1) | -0.007 (-1.1) | +0.007 (+1.2) | 0.542 | 0 |
| combo | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.003 | +0.004 (+0.6) | -0.007 (-1.0) | +0.011 (+1.7) | 0.592 | 0 |
| combo | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.002 | -0.003 (-0.5) | -0.007 (-0.9) | +0.003 (+0.6) | nan | 0 |
| emb_lr=0.01+softplus | 300 | 0.2 | 101 |  | rolling | 0.117 | +0.000 | +0.005 (+0.9) | +0.009 (+1.5) | -0.004 (-0.6) | 0.516 | 0 |
| emb_lr=0.01+softplus | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.007 | +0.011 (+2.0) | +0.015 (+2.6) | -0.004 (-0.6) | 0.517 | 0 |
| emb_lr=0.01+softplus | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.011 | +0.015 (+2.4) | +0.019 (+3.0) | -0.003 (-0.6) | 0.571 | 2 |
| emb_lr=0.01+softplus | 300 | 0.8 | 101 | null | rolling | 0.000 | +0.004 | +0.008 (+1.5) | +0.011 (+1.8) | -0.003 (-0.5) | nan | 0 |
| emb_lr=0.01+softplus | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.007 | +0.003 (+0.4) | +0.002 (+0.2) | +0.001 (+0.2) | 0.541 | 0 |
| emb_lr=0.01+softplus | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.005 | +0.005 (+0.8) | -0.003 (-0.4) | +0.008 (+1.3) | 0.547 | 0 |
| emb_lr=0.01+softplus | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.022 | +0.020 (+3.3) | +0.013 (+2.0) | +0.007 (+1.3) | 0.615 | 0 |
| emb_lr=0.01+softplus | 300 | 0.8 | 102 | null | rolling | 0.000 | +0.011 | +0.004 (+0.7) | +0.006 (+0.8) | -0.001 (-0.2) | nan | 0 |
| combo+softplus | 300 | 0.2 | 101 |  | rolling | 0.117 | +0.001 | +0.006 (+1.1) | +0.008 (+1.2) | -0.001 (-0.3) | 0.497 | 1 |
| combo+softplus | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.007 | +0.013 (+2.0) | +0.014 (+2.1) | -0.001 (-0.2) | 0.473 | 2 |
| combo+softplus | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.009 | +0.014 (+2.2) | +0.020 (+3.3) | -0.006 (-1.1) | 0.494 | 2 |
| combo+softplus | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.002 | +0.005 (+1.0) | +0.004 (+0.7) | +0.001 (+0.2) | nan | 0 |
| combo+softplus | 300 | 0.2 | 102 |  | rolling | 0.122 | -0.001 | -0.006 (-0.9) | -0.009 (-1.3) | +0.003 (+0.6) | 0.500 | 0 |
| combo+softplus | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.003 | +0.002 (+0.3) | -0.006 (-0.8) | +0.007 (+1.2) | 0.507 | 0 |
| combo+softplus | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.004 | +0.005 (+0.7) | -0.006 (-0.8) | +0.011 (+1.7) | 0.525 | 0 |
| combo+softplus | 300 | 0.8 | 102 | null | rolling | 0.000 | -0.002 | -0.007 (-1.1) | -0.011 (-1.5) | +0.003 (+0.6) | nan | 1 |
| prev | 300 | 0.2 | 101 |  | rolling | 0.117 | +0.003 | +0.008 (+1.4) | +0.012 (+2.0) | -0.003 (-0.6) | 0.553 | 0 |
| prev | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.010 | +0.016 (+2.3) | +0.018 (+3.0) | -0.002 (-0.3) | 0.651 | 2 |
| prev | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.050 | +0.055 (+7.9) | +0.055 (+8.3) | +0.000 (+0.0) | 0.748 | 2 |
| prev | 300 | 0.8 | 101 | null | rolling | 0.000 | +0.002 | +0.010 (+1.5) | +0.009 (+1.5) | +0.001 (+0.2) | nan | 2 |
| prev | 300 | 0.2 | 102 |  | rolling | 0.122 | -0.002 | -0.007 (-1.1) | -0.011 (-1.7) | +0.005 (+0.8) | 0.529 | 0 |
| prev | 300 | 0.3 | 102 |  | rolling | 0.182 | -0.001 | -0.002 (-0.3) | -0.007 (-0.9) | +0.005 (+0.9) | 0.586 | 0 |
| prev | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.011 | +0.012 (+1.9) | +0.005 (+0.6) | +0.007 (+1.2) | 0.667 | 0 |
| prev | 300 | 0.8 | 102 | null | rolling | 0.000 | -0.005 | -0.011 (-1.8) | -0.014 (-2.2) | +0.003 (+0.6) | nan | 0 |
| prev+softplus | 300 | 0.2 | 101 |  | rolling | 0.117 | -0.003 | +0.002 (+0.3) | +0.006 (+0.9) | -0.003 (-0.6) | 0.533 | 0 |
| prev+softplus | 300 | 0.3 | 101 |  | rolling | 0.177 | +0.017 | +0.023 (+3.0) | +0.024 (+3.5) | -0.002 (-0.3) | 0.640 | 2 |
| prev+softplus | 300 | 0.4 | 101 |  | rolling | 0.238 | +0.053 | +0.058 (+8.4) | +0.058 (+8.6) | +0.000 (+0.0) | 0.719 | 2 |
| prev+softplus | 300 | 0.8 | 101 | null | rolling | 0.000 | -0.005 | +0.003 (+0.4) | +0.002 (+0.3) | +0.001 (+0.2) | nan | 0 |
| prev+softplus | 300 | 0.2 | 102 |  | rolling | 0.122 | +0.001 | -0.004 (-0.6) | -0.008 (-1.3) | +0.005 (+0.8) | 0.505 | 0 |
| prev+softplus | 300 | 0.3 | 102 |  | rolling | 0.182 | +0.003 | +0.002 (+0.3) | -0.003 (-0.4) | +0.005 (+0.9) | 0.632 | 0 |
| prev+softplus | 300 | 0.4 | 102 |  | rolling | 0.243 | +0.021 | +0.022 (+3.4) | +0.014 (+2.0) | +0.007 (+1.2) | 0.737 | 0 |
| prev+softplus | 300 | 0.8 | 102 | null | rolling | 0.000 | -0.007 | -0.013 (-2.0) | -0.016 (-2.4) | +0.003 (+0.6) | nan | 0 |
