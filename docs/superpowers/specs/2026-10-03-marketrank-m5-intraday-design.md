# MarketRank M5: the intraday heartbeat and learned connectivity on a prior

**Status:** user-approved direction, 2026-10-03.

**Why.** M4 showed that learning connectivity from scratch on weekly data is underpowered. Reliable detection needs an oracle IC of about 0.24, against roughly 0.01–0.05 in real markets (`research/graphlearn/POWER.md`). The user's view is that the market's heartbeat is intraday: flows appear and fade within hours. Two changes follow:
- **Tighter time steps:** 15-minute bars give about 26 times more samples per day than daily bars.
- **Learn corrections, not a whole graph:** learn corrections to MarketRank's own estimated transition matrix instead of a free graph. That cuts the learned parameters from about 320k to a few thousand.

The Markov transition-probability graph and its solver stay the core. M5 improves the connectivity fed into the chain.

## User decisions (2026-10-03)
- **Bars:** 15 minutes, regular session only (09:30–16:00 ET, 26 bars per day).
- **Universe:** the 1,000 most liquid US stocks and ETFs, ranked by median daily dollar volume over 2024-10..2026-10 from the daily lake. This is a static list, so survivorship bias is acknowledged.
- **History:** 2 years, 2024-10-01 to 2026-10-02.
- **Target:** the forward return over the next 6 bars (1.5 hours), from the next bar's open. It is masked when the window crosses the session close, so no overnight returns.
- **Alpaca download:** approved, once, for this scope. `.env` keys are never printed, rate limits are respected, and the lake catalog is backed up first. The daily `tf=1d` data is untouched.

## Design
1. **Data (C++).** Add a `Timeframe::Min15` ("15m" → Alpaca "15Min") and a new lake partition `tf=15m`.
   - `--sync-intraday` takes a ticker list and a date range, then fetches with pagination and resume.
   - Session filtering covers holidays and half days.
   - The panel is built with a session calendar, so bar indices run within sessions and overnight gaps are explicit.
2. **Pipeline at 15 minutes.** Run the same CorePipeline (marketrank preset) on the 15m panel.
   - Bar-count windows are rescaled to keep the same wall-clock meaning: ADV 20 days = 520 bars; correlation window 60 days → 26×20 bars (documented choice).
   - The flux accumulator's half-life is in bars: the cumulative chain keeps its long memory, and a "heartbeat" chain uses a half-life of 26 bars (one day).
   - **Export:**
     - per-bar features;
     - eligibility;
     - the 6-bar label (masked across the close);
     - for each decision bar, the **prior edges**: MarketRank's kept transition edges (top k_out per node, with P_ji and raw dollars), in a compact edge file.
3. **Model (Python): prior-anchored connectivity.**
   - **Learned:** A_t = row-softmax over the prior's support of (log P_prior + g(features of i, j) + u_iᵀv_j). Here g is a small shared MLP over pair features (prior weight, return correlation, sector match, size ratio) and u, v have rank 4.
   - **Baselines:**
     - B0: no message.
     - B0E: per-stock embedding, no message.
     - **Bprior:** A = P_prior, no learning. This is the MarketRank chain as is.
     - Learned must beat **Bprior** as well as B0 and B0E. That is the M5 question: does learning improve the chain's connectivity?
   - Training uses daily-chunked samples (all 26 bars of a session form one batch), validation-IC early stopping with a minimum epoch count, restarts and signed messages, carried over from M4 round 3.
4. **Power first, then real data.**
   - Build a synthetic intraday market with a planted connectivity correction on a known prior, at realistic oracle ICs of 0.02, 0.03, 0.05 and 0.08.
   - Selection rule, committed before the grid: detect in at least 10 of 12 markets at oracle IC 0.05, with clean nulls. If no configuration passes, stop and report.
5. **Evaluation (pre-registered, real data).** Walk forward over the 2 years:
   - expanding window, retrain monthly, embargo of one session, rolling mode as the deployed regime.
   - **Primary:** the paired per-bar IC of learned − Bprior and of learned − B0E, with t > 2 by day-clustered standard errors, positive in at least 2/3 of months.
   - **Secondary:** a long-short spread of the top versus bottom decile over the 6-bar horizon after 5 bps per side, reported but not gated.
   - Every configuration is registered as a trial.

## Non-goals
Live trading, order execution, and extended-hours bars. The daily product and UI are unchanged in M5. An intraday heartbeat view in the UI is a later step, once the learned chain earns it.
