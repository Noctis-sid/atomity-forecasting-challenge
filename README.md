# Cloud Workload Forecasting: Azure VM trace (data-v1)

This repo is my submission for the Atomity forecasting take-home. It forecasts two fleet-level targets for the final two hours of the curated Azure Public Dataset V2 trace: `active_vm_count` and `aggregate_cpu_utilization`. It also turns the forecast into a capacity-headroom policy.

- **Report:** [`reports/final_report.md`](reports/final_report.md)
- **Analysis notebook:** [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb). It produces every number and figure in the report.
- **Submission file:** `predictions.csv`

## 1. Install

You need Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"       # numpy, pandas, scikit-learn, pytest
python -m pip install -e ".[analysis]"  # optional: matplotlib + jupyter for the notebook
python -m pytest -q
```

## 2. Data

Download `data-v1.tar.gz` from the [data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract it into `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
(cd challenge_data && sha256sum -c SHA256SUMS)
```

The data files are git-ignored and are never committed.

## 3. Generate the predictions

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This command does the following, in order:

1. Load the data and validate it.
2. Build both targets on a 5-minute grid.
3. Split the series by time: the first 110 slots are training, the last 24 are the holdout.
4. Run a rolling-origin backtest of every candidate model inside the training window.
5. Select a model for each target by backtest MAE.
6. Refit on all 110 training slots and forecast the 24 holdout slots.
7. Write `predictions.csv` and check it against the `SUBMISSION.md` contract.

On a laptop it takes about 3–4 minutes. Most of that time goes to the VM-level ML candidates. Add `--skip-panel` to run in about 10 seconds with the statistical models only. For this dataset that gives the same submission, because the statistical model wins the backtest.

It also writes these extra files to `artifacts/` (git-ignored):

- `prediction_intervals.csv`: 5/10/50/90/95% bands.
- `backtest_summary_<target>.csv` and `backtest_errors_<target>.csv`.
- `run_metadata.json`: which model was selected and why, plus the data-quality report.

You score the forecast against the holdout in a separate step, after `predictions.csv` exists:

```bash
python -m cwfc.evaluate --data challenge_data --predictions predictions.csv
```

To rebuild the notebook and the report figures:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

The pipeline doesn't depend on the exact files supplied. It handles a different number of VMs or timestamps, missing grid slots, duplicate or invalid readings, VMs that start or stop mid-trace, VMs with no metadata row, and a missing metadata file. `tests/test_pipeline.py` runs the CLI on a synthetic dataset with all of these problems.

## 4. Main modelling decisions

| Decision | Choice | Why |
|---|---|---|
| Cadence | Native 5 minutes | The data has no gaps, so there's no reason to resample. The holdout maps one-to-one to the grid. |
| `active_vm_count` | Distinct VMs with a finite `avg_cpu` in [0, 100] at each slot. Duplicates are collapsed. | Required by the contract. In data-v1 it's 10,000 at every slot, because the release only kept VMs with complete traces (see the report). |
| `aggregate_cpu_utilization` | Unweighted mean of valid `avg_cpu` per slot. An empty slot is NaN and is never imputed as a target. Model inputs forward-fill from the past only. | Follows the definition in the contract. |
| Holdout | The final 24 observed timestamps. | Required. |
| Validation | 16 expanding-window origins between slot 48 and slot 86, each forecasting 24 steps. All of them fall inside the 110 training slots. | Same task as production: fit on everything so far, forecast the next 2 hours. |
| Baselines | Persistence, moving averages (6/12/24), seasonal-naive with a 1-hour season, and drift. | Every one goes through the same backtest. |
| Candidates | `seasonal_profile`, `ses_profile`, and VM-level ML models (`panel_ridge`, `panel_gbm`, plus an ensemble). The ML models pool all 10k VMs and aggregate their forecasts bottom-up. | The aggregate series has only 110 points, but the 10k VM series behind it give the ML models about 1.5M training rows. |
| Selection | Lowest backtest MAE. If two models are within 1%, the simpler one wins. | Decided before the holdout was looked at, and the holdout is never used. |
| Selected (CPU) | **`ses_profile`**: an exponentially smoothed level plus a within-hour profile. The smoothing weight is fit in-sample on the history. | 15% lower backtest MAE than persistence, and it beats persistence on 62% of origins. |
| Selected (count) | Persistence | Every candidate scores exactly 0 error, so the simplest model is chosen. |
| Intervals | Empirical quantiles of the backtest residuals at each step. The bands are made monotone so they never narrow as the lead time grows. | Feeds the capacity policy. |
| Features excluded | The metadata columns `avg_cpu`, `max_cpu`, `p95_max_cpu` and `timestamp_vm_deleted` | They summarise each VM's whole 30-day life, so they contain future information. The metadata `avg_cpu` correlates 0.87 with the VM's mean over this window. |

## 5. Layout

```text
src/cwfc/
  data.py        load, clean, build targets on a regular grid, chronological split
  baselines.py   persistence, moving average, drift, seasonal-naive, seasonal_profile, ses_profile
  panel.py       VM-level pooled ML model (ridge / gradient boosting), aggregated bottom-up
  models.py      model registry; History = the training-only view each model receives
  backtest.py    rolling-origin backtest, metrics summary, model selection, residual quantiles
  policy.py      forecast + quantile headroom policy and its evaluation
  metrics.py     MAE, RMSE, sMAPE, WAPE, normalised MAE, interval coverage
  predict.py     CLI that writes predictions.csv
  evaluate.py    holdout scoring (reporting only; not imported by predict.py)
notebooks/analysis.ipynb
reports/final_report.md, reports/figures/
tests/           unit tests + an end-to-end smoke run + a leakage test
```

## 6. How leakage is prevented

- After the split, models only ever receive a `History` object. It holds the training-window targets and a VM matrix cut off at the last training timestamp. `predict.py` asserts this.
- Every backtest origin uses only the slots before it. Every panel-model training row has its target strictly before the origin.
- `tests/test_pipeline.py::test_holdout_values_cannot_influence_predictions` scrambles all the readings in the holdout window, and also drops 30% of them. It then checks that the predictions are byte-for-byte the same.
- Phase-of-hour features come from timestamps, which are known in advance for every slot being forecast. They are computed relative to the trace start, because the absolute time of day isn't published.
