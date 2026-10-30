# Case study 4: trading edge or luck? (moving-average crossovers)

**Question:** search 272 long/flat moving-average rules and keep the best one. Is its
out-performance over buy-and-hold an *edge*, or what the best of 272 rules with no skill would
show by luck?

**Run it**

```bash
uv run python case_studies/04_trading_edge/run.py                     # two synthetic worlds, ~1 min
uv run python case_studies/04_trading_edge/run.py --data prices.csv   # real daily prices (date, close)
uv run isbetter run case_studies/04_trading_edge/config.yaml          # the standard test, alone
```

Output: [`REPORT.md`](REPORT.md) (full write-up), `out/trials_*.jsonl` (the trial registry of
every variant evaluated) and `out/returns_*.csv` (the best variant's and buy-and-hold's daily
returns, which the `isbetter run` command above analyses). Peak memory is about 0.5 GB.

**Data caveat.** Real ETF prices could not be downloaded from the build environment, so the
committed report uses two **synthetic** daily-price worlds (10 years each): a *luck* world where
returns are unpredictable and no rule has an edge, and a *trend* world with persistent drifts
where trend-following has a real edge. They demonstrate the method, not the behaviour of real
markets. `--data` accepts a CSV with `date` and `close` (or `adj_close`) columns and runs the same
analysis on one real series; that path is tested only on a tiny file.

**What the report contrasts, for the best variant by Sharpe ratio:**

| tool | what it answers |
|---|---|
| naive t-test | is the mean excess return zero? (valid for one rule chosen in advance) |
| PSR | P(true Sharpe > 0), allowing for skew and fat tails, but still ignoring the search |
| **DSR** | P(true Sharpe > what the best of N unskilled variants reaches), N and Sharpe spread from the `TrialRegistry` |
| MinTRL | how long a track record would be needed |
| block-bootstrap CI | Sharpe interval that survives autocorrelation and volatility clustering |
| SPA / StepM | does any rule (which rules) beat buy-and-hold, accounting for the search |
| **PBO** | does picking the in-sample best parameters help out of sample? |

What is tested is the *excess return over buy-and-hold*, so a rule that is merely invested in a
rising market gets no credit. The report also repeats the experiment over 100 fresh worlds of
each kind to show how often each tool is fooled: in luck worlds the naive test flags the best
variant about 13% of the time against about 1% for DSR; in trend worlds DSR finds the real edge
less often (about 28% against 75%), which is the power cost of not being fooled.

**Not modelled:** transaction costs and slippage, short positions, position sizing.
