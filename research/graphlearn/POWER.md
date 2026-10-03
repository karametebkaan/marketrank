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
