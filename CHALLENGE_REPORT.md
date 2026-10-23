# Challenge Report: How I Approached the Atomity Forecasting Challenge

**Candidate:** Sidhant Pujari · **Repo:** https://github.com/Noctis-sid/atomity-forecasting-challenge · **Dates:** 21–23 October 2026

This document explains *how* I got to the solution: the order I worked in, the decisions and dead ends along the way, how I checked the result, and how I used AI tools. The technical results are in [`reports/final_report.md`](reports/final_report.md).

---

## 1. Reading the brief

Before writing any code I went through `ASSIGNMENT.md`, `DATASET.md` and `SUBMISSION.md`, and pulled out four constraints that would shape everything else:

1. **The holdout is public.** Nothing technical stops you from cheating, so the methodology has to *prove* there's no leakage, not just claim it.
2. **The scoring rewards understanding over model complexity.** About 70% of the marks are for data understanding, analysis, validation and error analysis. "Model quality" is only 15%, and the brief says an LSTM earns no extra points.
3. **Reviewers may run the pipeline on their own small "smoke" dataset.** So nothing could be hard-coded to these files.
4. **The starter code didn't fit the problem.** The validation helper assumed 7 days of history and a 1-day horizon. We have 110 points and a 24-point horizon.

## 2. First look at the data, and what it changed

I downloaded the `data-v1` release, verified the checksums and profiled it before building any model. Three findings changed my plan:

- **`active_vm_count` is always 10,000.** Every VM has all 134 readings, because the release only kept VMs with complete traces. So one of the two required targets is trivial by construction. I decided to say this plainly rather than present a perfect score as a modelling achievement.
- **The metadata contains future information.** The `avg_cpu`, `max_cpu` and `p95_max_cpu` columns summarise each VM's whole 30-day life. The metadata `avg_cpu` correlates 0.87 with the VM's mean over this window. Using them as features would be leakage, so I excluded them.
- **The aggregate series is very stable** (CV ≈ 2%). It has a strong hourly pattern (lag-12 autocorrelation 0.71) and one 1.3-hour surge. That pointed toward "level + hourly profile" models and meant the main risk would be level shifts, not noise.

## 3. How the solution evolved

I worked baseline-first and let the backtest decide at each step.

| Step | What I tried | What happened | Decision |
|---|---|---|---|
| 1 | Baselines: persistence, moving averages, seasonal-naive, drift | Persistence was hard to beat. Seasonal-naive was *worse*, because copying last hour also copies its noise. | Keep persistence as the reference baseline |
| 2 | Moving-average level plus an averaged within-hour profile | The hourly shape helped, but a fixed window lagged behind level changes | Need a smarter level estimate |
| 3 | **Exponential smoothing level + hourly profile** (`ses_profile`) | 15% better than persistence over 16 rolling backtests | Became the main model |
| 4 | ML: one gradient-boosting model trained across all 10,000 VMs (1.5M rows), then averaged up to the fleet | 16% better *per VM*, but no better for the fleet total | Kept as a documented negative result (see below) |
| 5 | Added per-VM EWM features to the ML model | Small improvement, still behind `ses_profile` | Stopped tuning; the limit is structural |
| 6 | Diagnosed *why* the errors happen | 73% of the error is shared level shifts, like the batch surge driven by 1% of VMs | Shifted effort from better point forecasts to a capacity policy that copes with those misses |
| 7 | Uncertainty bands from backtest residuals | They narrowed at long horizons, a small-sample artefact | Made the bands monotone in lead time |
| 8 | Capacity policy: forecast + residual quantile, with the quantile set by the cost of a shortfall (newsvendor rule) | Q0.90 had zero shortfall on the holdout with a third of the headroom of "provision to the max" | Became the recommendation |

**Why the ML model lost.** This was the most interesting lesson. The VM-level model really did learn something: each VM's mean reversion. But that is idiosyncratic noise, and it already cancels when you average 10,000 VMs. The fleet-level error comes from *correlated* shifts, such as one batch job hitting many VMs at once, and no single VM's history predicts those. I chose to report this honestly rather than force an ML model into the submission.

**Selection discipline.** The model was chosen only from the 16 rolling backtests inside the 110 training points. When I later scored all models on the holdout, a simpler variant ranked higher there. I didn't switch, because that would have been holdout-based model selection, and the report says so explicitly.

## 4. How I verified the work

- **Leakage test:** a unit test scrambles every holdout reading, drops 30% of them, and asserts the predictions are byte-for-byte identical.
- **Robustness test:** the full CLI runs on a synthetic dataset with gaps, duplicates, invalid readings, VMs starting and stopping mid-trace, and missing metadata.
- **Clean-checkout run** on Python 3.11: fresh environment, tests, prediction. The output was identical to the submitted file.
- **Local run on my own machine** (Windows, PyCharm, Python 3.14):
  - 12/12 tests passed.
  - `cwfc.predict` regenerated `predictions.csv` and `fc` reported *no differences*.
  - `cwfc.evaluate` reproduced the report's numbers exactly (MAE 0.1396, WAPE 1.51%).
- **Repo hygiene before pushing:**
  - I found that the working folder contained leftover scripts and figures from earlier exploration that the final pipeline didn't use. I moved them out and amended the commit so the repo has only what the report refers to.
  - I made sure `.venv`, the 116 MB dataset, `artifacts/` and IDE settings were all git-ignored.
  - I confirmed the public repo clones and passes its tests without signing in.

## 5. How I used AI tools

I used **Claude**, Anthropic's AI agent (Claude Code / Cowork), as a pair-programmer and agent throughout. I'd rather be explicit about this, because I think using AI agents well is part of the skill set for an AI/ML engineering role.

**How the agent workflow ran.**
- I gave Claude the challenge folder and the brief.
- It worked as an autonomous agent in a sandboxed environment: it downloaded and verified the data, ran exploratory analysis, wrote the `cwfc` package, ran backtests and experiments, executed the notebook, and drafted the report and README.
- It kept a task list and iterated in loops: run an experiment, read the result, change the approach. Steps 4–7 in the table above were iterations of that loop.
- I steered at each stage with short instructions. When I asked it to commit and push, it pushed back: the only remote was Atomity's own repo, so it committed locally and asked where it should go instead.

**What I did myself.**
- Directed the work and decided what to submit and how.
- Reviewed the approach, findings and trade-offs, including keeping the ML result as a negative finding and not switching models after seeing the holdout.
- Set up and ran the whole pipeline independently on my own machine to confirm it reproduces exactly.
- Handled the GitHub setup, fixed the remote configuration, and cleaned unused files out of the commit.
- Took ownership of the final submission. I can explain every design decision in it.

**What the AI did well.** Fast iteration: running a 16-fold backtest across 11 models, diagnosing *why* the ML model underperformed, and producing tests and documentation alongside the code instead of afterwards.

**Where human judgement mattered.** Setting the priorities in the brief (honest reporting over a flattering metric), checking the work on a real machine, spotting the stray files in the repo, and deciding what to submit.

Using an agent compressed what would normally be several days of coding into about three days of directing, reviewing and verifying. The methodology choices, and the responsibility for them, are mine.

## 6. What I'd do next with more time

- Use the full 30-day Azure trace to capture daily and weekly seasonality, and to model real VM churn.
- Detect surges early by watching deployment-level aggregates, where a batch job is a large, clear signal before it's diluted across 10,000 VMs.
- Forecast `max_cpu` (within-interval peaks) for host-level packing decisions.
- Add the optional cross-cloud comparison with Google ClusterData2019.
