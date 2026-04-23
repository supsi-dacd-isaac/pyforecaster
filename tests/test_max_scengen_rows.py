"""
Tests for optional post-fit row cap (``max_scengen_rows``) on ``ScenarioGenerator``.
"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from pyforecaster.forecaster import LinearForecaster, ScenarioGenerator
from pyforecaster.scenarios_generator import ScenGen
from pyforecaster.forecasting_models.fast_adaptive_models import Fourier_es
from pyforecaster.forecasting_models.holtwinters import HoltWinters, HoltWintersMulti
from pyforecaster.forecasting_models.randomforests import QRF


def _synth_xy(n, freq="h", seed=0, n_targets=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    x = pd.DataFrame(rng.standard_normal((n, 2)), index=idx, columns=["a", "b"])
    y = x.values[:, 0:1] * 2.0
    y = np.hstack([y, y + 0.05 * rng.standard_normal((n, 1))]) if n_targets > 1 else y
    y = y + 0.1 * rng.standard_normal((n, n_targets))
    cols = [f"t{i}" for i in range(n_targets)]
    y = pd.DataFrame(y, index=idx, columns=cols)
    return x, y


def _synth_signal_df(n, freq="h", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    base = np.sin(np.arange(n) * 2 * np.pi / 24)
    noise = 0.05 * rng.standard_normal(n)
    target = base + noise
    exog = np.cos(np.arange(n) * 2 * np.pi / 24) + 0.05 * rng.standard_normal(n)
    return pd.DataFrame({"target": target, "exog": exog}, index=idx)


class TestMaxScengenRowsValidation(unittest.TestCase):
    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            ScenarioGenerator(max_scengen_rows=0)
        with self.assertRaises(ValueError):
            ScenarioGenerator(max_scengen_rows=-1)

    def test_rejects_wrong_type(self):
        with self.assertRaises(TypeError):
            ScenarioGenerator(max_scengen_rows="nope")


class TestMaxScengenRowsBehavior(unittest.TestCase):
    def test_none_unchanged_vs_huge_cap(self):
        n = 60
        x, y = _synth_xy(n, seed=2)
        f1 = LinearForecaster(
            val_ratio=None, kind="linear", max_scengen_rows=None, scengen_random_state=0
        ).fit(x, y)
        f2 = LinearForecaster(
            val_ratio=None, kind="linear", max_scengen_rows=10_000, scengen_random_state=0
        ).fit(x, y)
        for h in f1.err_distr:
            np.testing.assert_allclose(
                f1.err_distr[h], f2.err_distr[h], rtol=0, atol=0, err_msg=f"hour {h}"
            )

    def test_subsampled_rows_reach_scengen_fit(self):
        n = 500
        x, y = _synth_xy(n, seed=3)
        cap = 100
        rows = []

        _orig = ScenGen.fit

        def capture_fit(slf, err_df, *args, **kwargs):
            rows.append(err_df.shape[0])
            return _orig(slf, err_df, *args, **kwargs)

        with patch.object(ScenGen, "fit", new=capture_fit):
            LinearForecaster(
                val_ratio=None,
                kind="linear",
                max_scengen_rows=cap,
                scengen_random_state=42,
            ).fit(x, y)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], cap)

    def test_x_y_stay_index_aligned(self):
        n = 200
        x, y = _synth_xy(n, seed=4)
        g = ScenarioGenerator(
            max_scengen_rows=30,
            scengen_random_state=7,
        )
        y2 = g.anti_transform(x, y)
        xs, ys = g._maybe_subsample_for_scengen(x, y2)
        self.assertTrue(xs.index.equals(ys.index))
        self.assertEqual(len(xs), 30)
        self.assertEqual(len(ys), 30)

    def test_reproducible_subsample(self):
        n = 200
        x, y = _synth_xy(n, seed=5)
        y0 = y.copy()
        f = LinearForecaster(
            val_ratio=None, kind="linear", max_scengen_rows=40, scengen_random_state=123
        )
        a = f._maybe_subsample_for_scengen(x, f.anti_transform(x, y0))[0]
        f2 = LinearForecaster(
            val_ratio=None, kind="linear", max_scengen_rows=40, scengen_random_state=123
        )
        b = f2._maybe_subsample_for_scengen(x, f2.anti_transform(x, y0))[0]
        pd.testing.assert_index_equal(a.index, b.index)

    def test_stratified_includes_several_hours(self):
        # Two weeks hourly => many hours; require variety in the sample
        n = 14 * 24
        x, y = _synth_xy(n, seed=6)
        f = LinearForecaster(
            val_ratio=None,
            kind="linear",
            max_scengen_rows=200,
            scengen_random_state=99,
        )
        y2 = f.anti_transform(x, y)
        xs, _ = f._maybe_subsample_for_scengen(x, y2)
        h = np.unique(xs.index.hour)
        self.assertGreater(len(h), 3)

    def test_qrf_predict_quantiles_runs(self):
        n = 100
        # Two columns + n_single=2 keeps QRF's predict stack in the 2+ target case (n_multistep=0).
        x, y = _synth_xy(n, seed=8, n_targets=2)
        t = int(0.8 * n)
        tr, te = x.iloc[:t], x.iloc[t:]
        y_tr, y_te = y.iloc[:t], y.iloc[t:]
        model = QRF(
            n_estimators=4,
            n_single=2,
            n_jobs=1,
            parallel=False,
            random_state=0,
            val_ratio=None,
            max_scengen_rows=25,
            scengen_random_state=0,
        )
        model.fit(tr, y_tr)
        pq = model.predict_quantiles(te, quantiles=[0.1, 0.5, 0.9])
        self.assertEqual(pq.shape[0], te.shape[0])

    def test_holtwinters_keeps_original_err_distr_shape(self):
        df = _synth_signal_df(10 * 24, seed=12)
        model = HoltWinters(
            periods=[24, 48],
            target_name="target",
            n_sa=4,
            optimize_hyperpars=False,
        ).fit(df)
        self.assertIsInstance(model.err_distr, np.ndarray)
        self.assertEqual(model.err_distr.shape, (4, len(model.q_vect)))

    def test_stateful_keeps_original_err_distr_shape(self):
        df = _synth_signal_df(10 * 24, seed=13)
        model = Fourier_es(
            target_name="target",
            n_sa=4,
            m=24,
            periodicity=4,
            optimize_hyperpars=False,
        ).fit(df)
        self.assertIsInstance(model.err_distr, np.ndarray)
        self.assertEqual(model.err_distr.shape, (4, len(model.q_vect)))

    def test_holtwinters_rejects_postfit_cap_kwargs(self):
        with self.assertRaises(TypeError):
            HoltWinters(
                periods=[24, 48],
                target_name="target",
                n_sa=4,
                optimize_hyperpars=False,
                max_scengen_rows=12,
            )

    def test_stateful_rejects_postfit_cap_kwargs(self):
        with self.assertRaises(TypeError):
            Fourier_es(
                target_name="target",
                n_sa=4,
                m=24,
                periodicity=4,
                optimize_hyperpars=False,
                max_scengen_rows=12,
            )

    def test_holtwinters_multi_rejects_postfit_cap_kwargs(self):
        with self.assertRaises(TypeError):
            HoltWintersMulti(
                periods=[24, 48],
                target_name="target",
                n_sa=4,
                models_periods=np.array([1, 4]),
                optimize_hyperpars=False,
                optimize_submodels_hyperpars=False,
                max_scengen_rows=12,
            )


if __name__ == "__main__":
    unittest.main()
