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
