"""Tests for vintage-aware Formatter transforms."""

from __future__ import annotations

import pickle
import resource
import time
import unittest

import numpy as np
import pandas as pd

from pyforecaster.formatter import Formatter, Transformer
from pyforecaster.vintage import (
    VintagePolicy,
    VintageTransformer,
    assert_causal_selection,
    materialize_vintage_features,
    normalize_vintage_frame,
    select_snapshot_codes,
)


def _make_vintage_long(
    snapshots,
    steps,
    signal="nwp.temperature",
    margin=pd.Timedelta("15min"),
    value_fn=None,
):
    rows = []
    for snap in snapshots:
        snap = pd.Timestamp(snap)
        if snap.tzinfo is None:
            snap = snap.tz_localize("UTC")
        else:
            snap = snap.tz_convert("UTC")
        for step in steps:
            vt = snap + pd.Timedelta(hours=int(step))
            value = value_fn(snap, step) if value_fn else float(snap.hour + step)
            rows.append(
                {
                    "snapshot_time": snap,
                    "available_at": snap + margin,
                    "valid_time": vt,
                    "signal": signal,
                    "value": np.float32(value),
                }
            )
    return pd.DataFrame(rows)


class TestVintageFormatter(unittest.TestCase):
    def setUp(self):
        self.dt = pd.Timedelta("1h")
        self.snapshots = pd.date_range("2024-01-01 00:00", periods=6, freq="6h", tz="UTC")
        self.steps = list(range(0, 13))
        self.vintage = _make_vintage_long(
            self.snapshots,
            self.steps,
            value_fn=lambda snap, step: float(snap.dayofyear * 100 + snap.hour + step),
        )
        self.origins = pd.date_range("2024-01-01 03:00", periods=24, freq="1h", tz="UTC")

    def test_same_valid_time_differs_by_origin(self):
        # Two snapshots covering the same valid time with different values.
        snap_a = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        snap_b = pd.Timestamp("2024-01-01 06:00", tz="UTC")
        valid = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        frame = pd.DataFrame(
            [
                {
                    "snapshot_time": snap_a,
                    "available_at": snap_a + pd.Timedelta("15min"),
                    "valid_time": valid,
                    "signal": "nwp.temperature",
                    "value": 1.0,
                },
                {
                    "snapshot_time": snap_b,
                    "available_at": snap_b + pd.Timedelta("15min"),
                    "valid_time": valid,
                    "signal": "nwp.temperature",
                    "value": 9.0,
                },
            ]
        )
        o1 = pd.Timestamp("2024-01-01 05:00", tz="UTC")
        o2 = pd.Timestamp("2024-01-01 07:00", tz="UTC")
        # o1 selects snap_a; target valid via lag -7 -> 12:00
        feat_o1 = VintageTransformer(["nwp.temperature"], lags=[-7], dt=self.dt).transform(
            pd.DatetimeIndex([o1]), frame
        )
        # o2 selects snap_b; target valid via lag -5 -> 12:00
        feat_o2 = VintageTransformer(["nwp.temperature"], lags=[-5], dt=self.dt).transform(
            pd.DatetimeIndex([o2]), frame
        )
        self.assertAlmostEqual(float(feat_o1.iloc[0, 0]), 1.0)
        self.assertAlmostEqual(float(feat_o2.iloc[0, 0]), 9.0)

        feats, prov = VintageTransformer(
            ["nwp.temperature"], lags=[-7, -5], dt=self.dt
        ).transform(pd.DatetimeIndex([o1, o2]), frame, return_provenance=True)
        self.assertAlmostEqual(float(feats.loc[o1, "nwp.temperature_lag_-007"]), 1.0)
        self.assertAlmostEqual(float(feats.loc[o2, "nwp.temperature_lag_-005"]), 9.0)
        self.assertLessEqual(prov.loc[o1, "available_at"], o1)
        self.assertLessEqual(prov.loc[o2, "available_at"], o2)

    def test_excludes_future_snapshots_and_15min_margin(self):
        snap = pd.Timestamp("2024-01-01 06:00", tz="UTC")
        frame = _make_vintage_long([snap], [0, 1, 2], margin=pd.Timedelta("15min"))
        # Origin before available_at must not see the snapshot.
        too_early = pd.DatetimeIndex([snap + pd.Timedelta("10min")])
        tr = VintageTransformer(["nwp.temperature"], lags=[0], dt=pd.Timedelta("15min"))
        # Align dt to 15min and ask for valid_time == snap via lag that lands on the grid.
        # Use hourly dt with origin on the hour after availability.
        tr = VintageTransformer(["nwp.temperature"], lags=[0], dt=self.dt)
        # Origin 10 minutes after snapshot is before available_at (snap+15min)
        feats = tr.transform(too_early, frame)
        self.assertTrue(np.isnan(feats.to_numpy()).all())

        # First hourly origin at/after available_at that exists on the valid_time grid:
        # snap+1h is after available_at and is a valid_time in the frame.
        ok = pd.DatetimeIndex([snap + pd.Timedelta("1h")])
        feats_ok = tr.transform(ok, frame)
        self.assertFalse(np.isnan(feats_ok.to_numpy()).all())
        # Exactly at available_at: snapshot is selectable, but valid_time grid is hourly;
        # lag 0 at snap+15min has no exact valid_time — still no future fill.
        at_margin = pd.DatetimeIndex([snap + pd.Timedelta("15min")])
        feats_margin = tr.transform(at_margin, frame)
        self.assertTrue(np.isnan(feats_margin.to_numpy()).all())
        # Provenance must still select the snapshot at the margin boundary.
        _, prov = tr.transform(at_margin, frame, return_provenance=True)
        self.assertEqual(prov.iloc[0]["snapshot_time"], snap)
    def test_coherent_run_and_fallback_without_future_fill(self):
        # Incomplete latest snapshot should fall back to previous complete one when
        # missing values remain NaN (no forward fill from a newer incomplete run).
        snap_old = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        snap_new = pd.Timestamp("2024-01-01 06:00", tz="UTC")
        rows = []
        for step in range(0, 6):
            rows.append(
                {
                    "snapshot_time": snap_old,
                    "available_at": snap_old + pd.Timedelta("15min"),
                    "valid_time": snap_old + pd.Timedelta(hours=step),
                    "signal": "nwp.temperature",
                    "value": 10.0 + step,
                }
            )
        # New snapshot only covers step 0-1
        for step in range(0, 2):
            rows.append(
                {
                    "snapshot_time": snap_new,
                    "available_at": snap_new + pd.Timedelta("15min"),
                    "valid_time": snap_new + pd.Timedelta(hours=step),
                    "signal": "nwp.temperature",
                    "value": 90.0 + step,
                }
            )
        frame = pd.DataFrame(rows)
        origin = pd.DatetimeIndex([snap_new + pd.Timedelta("1h")])
        # lag -4 asks for valid_time = origin + 4h = 11:00, present only in old run.
        # Coherent selection still uses the latest available snapshot (new), so value is NaN
        # rather than silently filling from a mixed run.
        tr = VintageTransformer(
            ["nwp.temperature"],
            lags=[-4],
            dt=self.dt,
            policy=VintagePolicy(fallback=True, assert_causal=True),
        )
        feats, prov = tr.transform(origin, frame, return_provenance=True)
        self.assertEqual(prov.iloc[0]["snapshot_time"], snap_new)
        self.assertTrue(np.isnan(feats.to_numpy()).all())

    def test_column_names_match_legacy_transformer(self):
        wide = (
            self.vintage.pivot_table(
                index="valid_time", columns="signal", values="value", aggfunc="last"
            )
            .sort_index()
            .asfreq("1h")
        )
        legacy = Transformer(["nwp.temperature"], lags=[0, -1, -2], dt=self.dt)
        legacy_out = legacy.transform(wide, augment=False)
        vintage_tr = VintageTransformer(["nwp.temperature"], lags=[0, -1, -2], dt=self.dt)
        vintage_tr._ensure_metadata(self.dt)
        self.assertEqual(list(legacy_out.columns), list(vintage_tr.generated_features))
        self.assertEqual(list(legacy.metadata.index), list(vintage_tr.metadata.index))

    def test_timezone_pickle_and_legacy_formatter_load(self):
        origins = self.origins.tz_convert("Europe/Zurich")
        formatter = (
            Formatter(augment=True, dt=self.dt)
            .add_transform(["sensor.load"], lags=[1, 2])
            .add_vintage_transform(["nwp.temperature"], lags=[0, -1])
            .add_target_transform(["sensor.load"], lags=[-1])
        )
        observed = pd.DataFrame(
            {"sensor.load": np.arange(len(origins), dtype=float)},
            index=origins,
        )
        x, y, prov = formatter.transform(
            observed,
            vintage_inputs=self.vintage,
            return_target=True,
            return_vintage_provenance=True,
            time_features=False,
        )
        self.assertIsNotNone(prov)
        self.assertTrue((prov["available_at"] <= prov.index.tz_convert("UTC")).all())

        payload = pickle.dumps(formatter)
        loaded = pickle.loads(payload)
        self.assertEqual(len(loaded.vintage_transformers), 1)

        # Legacy formatter pickle without vintage attrs still loads and transforms.
        legacy = Formatter(augment=True, dt=self.dt).add_transform(
            ["sensor.load"], lags=[1]
        )
        legacy.add_target_transform(["sensor.load"], lags=[-1])
        blob = pickle.dumps(legacy)
        # Simulate pre-vintage object by stripping attributes after load
        legacy_loaded = pickle.loads(blob)
        if hasattr(legacy_loaded, "vintage_transformers"):
            delattr(legacy_loaded, "vintage_transformers")
        x_legacy, _ = legacy_loaded.transform(observed, time_features=False)
        self.assertGreater(x_legacy.shape[1], 0)

    def test_serialization_excludes_prepared_weather_and_preserves_causality(self):
        tr = VintageTransformer(["nwp.temperature"], lags=[0, -1], dt=self.dt)
        expected, provenance = tr.transform(self.origins, self.vintage, return_provenance=True)
        original_lookup = tr._lookups["nwp.temperature"]
        # A large prepared training cache must never become a model artifact.
        original_lookup.unused_training_cache = np.zeros(1_000_000, dtype=np.float32)
        blob = pickle.dumps(tr)
        self.assertLess(len(blob), 20_000)
        self.assertIs(tr._lookups["nwp.temperature"], original_lookup)
        loaded = pickle.loads(blob)
        self.assertEqual(loaded._lookups, {})
        self.assertIsNone(loaded._snapshots)
        actual, actual_provenance = loaded.transform(
            self.origins, self.vintage, return_provenance=True
        )
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        pd.testing.assert_frame_equal(actual_provenance, provenance, check_exact=True)
        # Replaying earlier origins still excludes unavailable snapshots.
        self.assertTrue((actual_provenance["available_at"] <= self.origins).all())
        early = pd.DatetimeIndex([self.snapshots[0] + pd.Timedelta("10min")])
        self.assertTrue(loaded.transform(early, self.vintage).isna().all().all())

    def test_historical_replay_matches_one_origin_live(self):
        formatter = (
            Formatter(augment=False, dt=self.dt)
            .add_vintage_transform(["nwp.temperature"], lags=[0, -1, -2])
        )
        x_hist, _ = formatter.transform(
            pd.DataFrame(index=self.origins),
            vintage_inputs=self.vintage,
            return_target=False,
            time_features=False,
        )
        live_origin = self.origins[10:11]
        x_live, _ = formatter.transform(
            pd.DataFrame(index=live_origin),
            vintage_inputs=self.vintage,
            return_target=False,
            time_features=False,
        )
        pd.testing.assert_frame_equal(
            x_hist.loc[live_origin].sort_index(axis=1),
            x_live.sort_index(axis=1),
            check_dtype=False,
        )

    def test_growth_is_origins_by_features_not_runs_by_steps(self):
        n_origins = 200
        n_snaps = 40
        n_steps = 48
        snaps = pd.date_range("2024-01-01", periods=n_snaps, freq="6h", tz="UTC")
        vintage = _make_vintage_long(snaps, list(range(n_steps)))
        origins = pd.date_range(snaps[1] + pd.Timedelta("3h"), periods=n_origins, freq="1h", tz="UTC")
        lags = list(range(0, -13, -1))
        tr = VintageTransformer(["nwp.temperature"], lags=lags, dt=self.dt)

        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        feats = tr.transform(origins, vintage)
        elapsed = time.perf_counter() - t0
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_delta_kb = max(0, rss_after - rss_before)

        n_features = len(lags)
        self.assertEqual(feats.shape, (n_origins, n_features))
        # Structural: output cells == origins * final features (not origins*runs*steps)
        self.assertEqual(feats.size, n_origins * n_features)
        self.assertLess(feats.size, n_origins * n_snaps * n_steps / 10)
        # float32 payload should stay near origins*features*4 bytes (allow overhead)
        expected_bytes = n_origins * n_features * 4
        self.assertLess(feats.memory_usage(deep=True).sum(), expected_bytes * 20)
        print(
            f"vintage_microbench origins={n_origins} features={n_features} "
            f"elapsed_s={elapsed:.4f} peak_rss_delta_kb={peak_delta_kb} "
            f"frame_bytes={feats.memory_usage(deep=True).sum()}"
        )

    def test_normalize_requires_valid_time_or_step(self):
        bad = pd.DataFrame(
            {
                "snapshot_time": [pd.Timestamp("2024-01-01", tz="UTC")],
                "signal": ["nwp.temperature"],
                "value": [1.0],
            }
        )
        with self.assertRaises(ValueError):
            normalize_vintage_frame(bad)

    def test_causal_assertion(self):
        origins = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
        avail = pd.DatetimeIndex([pd.Timestamp("2024-01-01 01:00", tz="UTC")])
        with self.assertRaises(AssertionError):
            assert_causal_selection(origins, avail)


if __name__ == "__main__":
    unittest.main()
