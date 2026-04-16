![build and test](https://github.com/hivepower/pyforecaster/actions/workflows/python-app.yml/badge.svg)

# Pyforecaster

Python utilities for **time-series forecasting**, **probabilistic outputs** (quantiles and scenarios), and **feature engineering** on `pandas` data with a `DatetimeIndex`. Models share a common pattern: fit a point (or native probabilistic) predictor, then layer **residual uncertainty** and optional **scenario trees** for stochastic optimization and risk analysis.

## Requirements

- **Python** 3.10 or newer (see `setup.py`).
- Install from the repository root:

```bash
pip install -r requirements.txt
# or
pip install .
```

Heavy dependencies include **NumPy**, **pandas**, **scikit-learn**, **LightGBM**, **statsmodels**, **JAX** / **Flax** / **Optax**, **Keras** (used by the bidirectional LSTM forecaster), **Optuna**, **Ray**, **quantile-forest**, and others listed in `setup.py`.

### Optional: tree visualization

To use Graphviz-based tree visualization:

```bash
sudo apt-get install libgraphviz-dev graphviz
pip install pygraphviz
```

(`pygraphviz` is also available as the `graphviz` extra in `setup.py`.)

---

## Core ideas

### `ScenarioGenerator` (`pyforecaster.forecaster`)

Base class for forecasters that:

- **`fit` / `predict`** — point forecasts (multi-step, multi-target as columns).
- **`predict_quantiles`** — combines predictions with **residual quantiles** (often conditional on hour of day).
- **`predict_scenarios`** / **`predict_trees`** — sample coherent multi-step paths via **`ScenGen`** (copula + scenario tree / reduction).

Registered convenience names for some models live in **`FORECASTER_MAP`** (`pyforecaster.forecasters_dictionaries`).

### Feature pipeline (`pyforecaster.formatter`)

**`Formatter`** builds supervised learning matrices from raw series: calendar and holiday features, lagged transforms, aggregations, train/test splits aligned in time, optional target normalization (with inverse transforms for inference), and parallel helpers for large frames (`big_data_utils`).

### Scenario generation (`pyforecaster.scenarios_generator`, `pyforecaster.scenred`, `pyforecaster.copula`, `pyforecaster.tree_builders`)

**`ScenGen`** ties:

- **Copulas** (see `pyforecaster.dictionaries.COPULA_MAP`): e.g. `HourlyGaussianCopula`, `ConditionalGaussianCopula`.
- **Trees / reduction** (`TREE_MAP`): e.g. `ScenredTree`, `DiffTree`, `NeuralGas`, `QuantileTree`.

Used to go from predicted quantiles (or point forecasts) to **scenarios** or **tree structures** for control under uncertainty.

---

## Forecasting models (overview)

| Area | Module(s) | What it does |
|------|-----------|----------------|
| **Linear / ridge** | `forecaster.LinearForecaster` | sklearn linear or ridge regression per `ScenarioGenerator` workflow. |
| **LightGBM** | `forecaster.LGBForecaster` | One booster per target column; residual quantiles for uncertainty. |
| **LightGBM hybrid** | `forecasting_models.gradientboosters.LGBMHybrid` | Mix of short-horizon “single-step” models and a shared multi-step model (with optional feature pruning using the formatter). |
| **Quantile random forest** | `forecasting_models.randomforests.QRF` | `quantile_forest.RandomForestQuantileRegressor` with a similar single vs multi-step structure. |
| **Holt–Winters / state-space style** | `forecasting_models.holtwinters` | Classical smoothing with numba-backed pieces, multi-series variants, hyperparameter search helpers. |
| **Statsmodels** | `forecasting_models.statsmodels_wrapper` | Wrapper around statsmodels (e.g. exponential smoothing) with rolling updates and quantiles from residuals. |
| **Fast / adaptive / Fourier** | `forecasting_models.fast_adaptive_models`, `random_fourier_features` | Stateful forecasters (Kalman-style building blocks), basis expansions, etc. |
| **Benchmarks** | `forecasting_models.benchmarks` | Simple baselines: persistent, seasonal persistent, discrete empirical distributions. |
| **Identity** | `forecasting_models.identity.IdentityForecaster` | Returns features as “predictions” (debugging / plumbing). |
| **Neural (JAX / Flax)** | `forecasting_models.neural_models.base_nn`, `ICNN.py`, `INN.py` | **`NN`** base with **`FFNN`**, **`LSTMNN`**, partially input-convex nets (**`PICNN`** and variants: structured, latent, recurrent-stable, etc.), and **causal invertible** models (**`CausalInvertibleNN`**). Training uses **Optax** and optional probabilistic losses (e.g. CRPS-style terms). |
| **Neural (Keras)** | `forecasting_models.neural_models.LSTM_bidir` | Bidirectional **LSTM** plus an exogenous MLP branch with skip connections (`tensorflow.keras`). |

---

## Other tools

- **`pyforecaster.trainer`** — **`cross_validate`**, **`optuna_objective`**, **`hyperpar_optimizer`** (Optuna + time-based CV when a formatter is provided). Default search spaces for some models are in **`pyforecaster.dictionaries.HYPERPAR_MAP`**.
- **`pyforecaster.metrics`** — **`nmae`**, **`rmse`**, **`mape`**, **`quantile_scores`**, sklearn-compatible **`make_scorer`**.
- **`pyforecaster.reconciliation.reconciliation`** — **`HierarchicalReconciliation`** for reconciling base forecasts across a hierarchy (e.g. min trace–style ideas, with scenario options).
- **`pyforecaster.plot_utils`** — Plotting helpers (e.g. scenarios, animations).
- **`pyforecaster.utilities`**, **`stat_utils`**, **`preprocessing.outlier_detection`** — misc stats and preprocessing.
- **`pyforecaster.big_data_utils`** — memory reduction, Ray- and multiprocessing-oriented helpers for large `DataFrame`s.

Tests under `tests/` exercise formatters, models, scenarios, metrics, and trainers.

## License

See `LICENSE` (MIT).
