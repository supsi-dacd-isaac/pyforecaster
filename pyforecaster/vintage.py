"""Vintage-aware NWP feature materialization for Formatter.

Selects a coherent forecast run per origin using availability times and
produces feature columns identical to the legacy Transformer naming scheme,
without building an origins × runs × steps Pandas cross-join.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pyforecaster.utilities import get_logger

DEFAULT_AVAILABILITY_MARGIN = pd.Timedelta("15min")
DEFAULT_CHUNK_SIZE = 4096

REQUIRED_LONG_COLUMNS = {"snapshot_time", "signal", "value"}


@dataclass
class VintagePolicy:
    """Causal availability policy for forecast vintages."""

    mode: str = "latest_available"
    availability_margin: Union[str, pd.Timedelta] = DEFAULT_AVAILABILITY_MARGIN
    fallback: bool = True
    assert_causal: bool = True
    return_provenance: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self):
        self.mode = str(self.mode)
        if self.mode not in {"latest_available", "legacy_latest_valid"}:
            raise ValueError(
                f"Unsupported vintage mode {self.mode!r}; "
                "expected 'latest_available' or 'legacy_latest_valid'."
            )
        self.availability_margin = pd.Timedelta(self.availability_margin)
        self.chunk_size = int(self.chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["availability_margin"] = str(pd.Timedelta(self.availability_margin))
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "VintagePolicy":
        if payload is None:
            return cls()
        return cls(**payload)


def _ensure_utc_index(index: pd.DatetimeIndex, name: str = "index") -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _ensure_utc_series(values: pd.Series, name: str) -> pd.Series:
    series = pd.to_datetime(values, utc=True)
    if getattr(series.dt, "tz", None) is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    return series


def normalize_vintage_frame(
    frame: pd.DataFrame,
    availability_margin: Union[str, pd.Timedelta] = DEFAULT_AVAILABILITY_MARGIN,
    signal_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Normalize a long vintage frame to a canonical schema.

    Accepted columns:
      - snapshot_time (required)
      - available_at (optional; default snapshot_time + availability_margin)
      - valid_time or step (one required; valid_time = snapshot_time + step)
      - signal, value (required)
    """
    if frame is None or len(frame) == 0:
        raise ValueError("vintage frame is empty")

    df = frame.copy()
    missing = REQUIRED_LONG_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"vintage frame missing required columns: {sorted(missing)}")

    availability_margin = pd.Timedelta(availability_margin)
    df["snapshot_time"] = _ensure_utc_series(df["snapshot_time"], "snapshot_time")

    if "available_at" in df.columns:
        df["available_at"] = _ensure_utc_series(df["available_at"], "available_at")
    else:
        df["available_at"] = df["snapshot_time"] + availability_margin

    if "valid_time" in df.columns:
        df["valid_time"] = _ensure_utc_series(df["valid_time"], "valid_time")
    elif "step" in df.columns:
        steps = pd.to_timedelta(df["step"])
        df["valid_time"] = df["snapshot_time"] + steps
    else:
        raise ValueError("vintage frame must provide either 'valid_time' or 'step'")

    df["signal"] = df["signal"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["snapshot_time", "available_at", "valid_time", "signal", "value"])

    if signal_names is not None:
        df = df.loc[df["signal"].isin(list(signal_names))]

    # Keep a single value per (snapshot, signal, valid_time); last write wins.
    df = (
        df.sort_values(["snapshot_time", "signal", "valid_time"])
        .drop_duplicates(subset=["snapshot_time", "signal", "valid_time"], keep="last")
        .reset_index(drop=True)
    )
    return df[
        ["snapshot_time", "available_at", "valid_time", "signal", "value"]
    ]


def _encode_times(times: pd.DatetimeIndex) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    times = _ensure_utc_index(pd.DatetimeIndex(times))
    codes = times.asi8.astype(np.int64, copy=False)
    return codes, times


def _snapshot_table(vintage: pd.DataFrame) -> pd.DataFrame:
    snaps = (
        vintage.groupby("snapshot_time", sort=True)["available_at"]
        .max()
        .reset_index()
        .sort_values("available_at")
        .reset_index(drop=True)
    )
    snaps["snapshot_code"] = np.arange(len(snaps), dtype=np.int32)
    return snaps


def select_snapshot_codes(
    origins: pd.DatetimeIndex,
    snapshots: pd.DataFrame,
    mode: str = "latest_available",
    fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map each origin to a snapshot_code.

    Returns
    -------
    codes : int32 array, -1 when no causal snapshot exists
    available_ats : datetime64[ns, UTC] aligned to selected snapshots (NaT if missing)
    """
    origins = _ensure_utc_index(pd.DatetimeIndex(origins))
    if snapshots.empty:
        return np.full(len(origins), -1, dtype=np.int32), pd.DatetimeIndex(
            [pd.NaT] * len(origins), tz="UTC"
        )

    if mode == "legacy_latest_valid":
        # Leaky baseline: always use the chronologically latest snapshot that
        # covers the origin as a valid_time elsewhere; here approximated as
        # the latest snapshot overall (parity with drop_duplicates keep=first
        # by forecast_time when collapsing by valid_time). Callers that need
        # exact legacy frames should still use the legacy wide path.
        codes = np.full(len(origins), int(snapshots["snapshot_code"].iloc[-1]), dtype=np.int32)
        available = snapshots["available_at"].iloc[-1]
        return codes, pd.DatetimeIndex([available] * len(origins), tz="UTC")

    avail = snapshots["available_at"].to_numpy(dtype="datetime64[ns]")
    origin_ns = origins.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    # searchsorted right: last index with avail <= origin
    idx = np.searchsorted(avail, origin_ns, side="right") - 1
    if not fallback:
        # only accept exact latest; already the semantics of searchsorted
        pass
    codes = np.where(idx >= 0, snapshots["snapshot_code"].to_numpy()[np.maximum(idx, 0)], -1)
    codes = codes.astype(np.int32, copy=False)
    codes = np.where(idx >= 0, codes, np.int32(-1))
    selected_avail = pd.DatetimeIndex(
        np.where(idx >= 0, avail[np.maximum(idx, 0)], np.datetime64("NaT")),
        tz="UTC",
    )
    return codes, selected_avail


def assert_causal_selection(
    origins: pd.DatetimeIndex,
    selected_available_at: pd.DatetimeIndex,
    allow_missing: bool = True,
) -> None:
    origins = _ensure_utc_index(pd.DatetimeIndex(origins))
    selected_available_at = _ensure_utc_index(pd.DatetimeIndex(selected_available_at))
    mask = selected_available_at.notna()
    if not allow_missing and not bool(mask.all()):
        raise AssertionError("Missing snapshot selection for one or more origins")
    if mask.any():
        violations = selected_available_at[mask] > origins[mask]
        if bool(np.any(violations)):
            n = int(np.sum(violations))
            raise AssertionError(
                f"Causal vintage violation: {n} origins selected a snapshot with available_at > origin"
            )


def densify_vintage_frame(
    vintage: pd.DataFrame,
    dt: Union[str, pd.Timedelta],
) -> pd.DataFrame:
    """Resample each snapshot/signal series onto formatter dt with PCHIP interpolation."""
    from scipy.interpolate import PchipInterpolator

    dt = pd.Timedelta(dt)
    if vintage.empty:
        return vintage
    pieces = []
    for (snap, signal), group in vintage.groupby(["snapshot_time", "signal"], sort=False):
        series = (
            group.sort_values("valid_time")
            .drop_duplicates("valid_time", keep="last")
            .set_index("valid_time")["value"]
            .astype(np.float64)
            .dropna()
        )
        if series.empty:
            continue
        idx = pd.date_range(series.index.min(), series.index.max(), freq=dt, tz="UTC")
        x_ns = series.index.tz_convert("UTC").tz_localize(None).asi8.astype(np.float64)
        y = series.to_numpy(dtype=np.float64)
        target_ns = idx.tz_convert("UTC").tz_localize(None).asi8.astype(np.float64)
        if len(series) == 1:
            values = np.full(len(idx), y[0], dtype=np.float32)
        else:
            # PCHIP is shape-preserving and avoids overshoot between knots.
            interpolator = PchipInterpolator(x_ns, y, extrapolate=False)
            values = interpolator(target_ns).astype(np.float32)
        avail = group["available_at"].iloc[0]
        piece = pd.DataFrame(
            {
                "snapshot_time": snap,
                "available_at": avail,
                "valid_time": idx,
                "signal": signal,
                "value": values,
            }
        )
        piece["step"] = piece["valid_time"] - piece["snapshot_time"]
        pieces.append(piece)
    if not pieces:
        return vintage.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


class _SignalLookup:
    """Dense float32 lookup table for one signal: rows=snapshots, cols=valid_times."""

    def __init__(self, vintage: pd.DataFrame, signal: str, snapshots: pd.DataFrame):
        signal_df = vintage.loc[vintage["signal"] == signal]
        if signal_df.empty:
            self.valid_times = pd.DatetimeIndex([], tz="UTC")
            self.valid_codes = np.empty(0, dtype=np.int64)
            self.matrix = np.empty((len(snapshots), 0), dtype=np.float32)
            self.snap_code_to_row = {
                int(c): i for i, c in enumerate(snapshots["snapshot_code"].to_numpy())
            }
            return

        valid_times = pd.DatetimeIndex(sorted(signal_df["valid_time"].unique())).tz_convert("UTC")
        self.valid_times = valid_times
        self.valid_codes = valid_times.asi8.astype(np.int64, copy=False)
        self.snap_code_to_row = {
            int(c): i for i, c in enumerate(snapshots["snapshot_code"].to_numpy())
        }
        n_rows = len(snapshots)
        n_cols = len(valid_times)
        matrix = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

        snap_map = snapshots.set_index("snapshot_time")["snapshot_code"]
        rows = signal_df["snapshot_time"].map(snap_map).to_numpy(dtype=np.int32)
        cols = pd.DatetimeIndex(signal_df["valid_time"]).tz_convert("UTC").asi8
        col_idx = np.searchsorted(self.valid_codes, cols)
        # guard against mismatches
        ok = (col_idx < n_cols) & (self.valid_codes[np.minimum(col_idx, max(n_cols - 1, 0))] == cols)
        ok &= rows >= 0
        matrix[rows[ok], col_idx[ok]] = signal_df["value"].to_numpy(dtype=np.float32)[ok]
        self.matrix = matrix

    def gather(self, snapshot_codes: np.ndarray, valid_time_codes: np.ndarray) -> np.ndarray:
        out = np.full(len(snapshot_codes), np.nan, dtype=np.float32)
        if self.matrix.size == 0:
            return out
        valid_mask = snapshot_codes >= 0
        if not np.any(valid_mask):
            return out
        rows = np.array(
            [self.snap_code_to_row.get(int(c), -1) for c in snapshot_codes],
            dtype=np.int32,
        )
        col_idx = np.searchsorted(self.valid_codes, valid_time_codes)
        n_cols = len(self.valid_codes)
        col_ok = (col_idx < n_cols) & (
            self.valid_codes[np.minimum(col_idx, max(n_cols - 1, 0))] == valid_time_codes
        )
        ok = valid_mask & (rows >= 0) & col_ok
        out[ok] = self.matrix[rows[ok], col_idx[ok]]
        return out

    def gather_window(
        self,
        snapshot_code: int,
        start_ns: np.int64,
        end_ns: np.int64,
        agg_fun: Optional[str],
    ) -> float:
        if snapshot_code < 0 or self.matrix.size == 0:
            return np.float32(np.nan)
        row = self.snap_code_to_row.get(int(snapshot_code), -1)
        if row < 0:
            return np.float32(np.nan)
        left = np.searchsorted(self.valid_codes, start_ns, side="left")
        right = np.searchsorted(self.valid_codes, end_ns, side="left")
        if right <= left:
            return np.float32(np.nan)
        window = self.matrix[row, left:right]
        if np.all(np.isnan(window)):
            return np.float32(np.nan)
        if agg_fun in (None, "none"):
            # last non-nan in window (point sample at end convention)
            finite = window[~np.isnan(window)]
            return np.float32(finite[-1]) if len(finite) else np.float32(np.nan)
        if agg_fun == "mean":
            return np.float32(np.nanmean(window))
        if agg_fun == "max":
            return np.float32(np.nanmax(window))
        if agg_fun == "min":
            return np.float32(np.nanmin(window))
        if agg_fun == "sum":
            return np.float32(np.nansum(window))
        if agg_fun == "std":
            return np.float32(np.nanstd(window))
        raise ValueError(f"Unsupported aggregation function for vintage transform: {agg_fun}")


class VintageTransformer:
    """Mirror of Transformer metadata/naming with causal vintage materialization."""

    Anyarray = Union[tuple, np.ndarray, list, None]

    def __init__(
        self,
        names,
        functions: Anyarray = None,
        agg_freq: Union[str, int, None] = None,
        lags: Anyarray = None,
        logger=None,
        relative_lags: bool = False,
        agg_bins: Anyarray = None,
        nested: bool = True,
        dt=None,
        name=None,
        policy: Optional[VintagePolicy] = None,
    ):
        # Local import avoids circular dependency with formatter.Formatter.
        from pyforecaster.formatter import Transformer

        self.base = Transformer(
            names,
            functions=functions,
            agg_freq=agg_freq,
            lags=lags,
            logger=logger,
            relative_lags=relative_lags,
            agg_bins=agg_bins,
            nested=nested,
            dt=dt,
            name=name,
        )
        self.names = self.base.names
        self.functions = self.base.functions
        self.agg_freq = self.base.agg_freq
        self.lags = self.base.lags
        self.relative_lags = self.base.relative_lags
        self.logger = self.base.logger if logger is None else logger
        self.dt = dt
        self.name = name
        self.policy = policy or VintagePolicy()
        self.metadata = None
        self.generated_features = None
        self._lookups: Dict[str, _SignalLookup] = {}
        self._snapshots: Optional[pd.DataFrame] = None

    def _ensure_metadata(self, dt: pd.Timedelta) -> pd.DataFrame:
        if self.metadata is not None and self.generated_features is not None:
            return self.metadata
        # Simulate on a tiny dummy frame so naming/metadata match Transformer.
        idx = pd.date_range("2000-01-01", periods=max(16, 8), freq=dt, tz="UTC")
        dummy = pd.DataFrame(
            {n: np.arange(len(idx), dtype=float) for n in self.names},
            index=idx,
        )
        self.base.dt = dt
        _ = self.base.transform(dummy, augment=False, simulate=True)
        self.metadata = self.base.metadata.copy()
        self.generated_features = list(self.metadata.index)
        return self.metadata

    def prepare(
        self,
        vintage_frame: pd.DataFrame,
        policy: Optional[VintagePolicy] = None,
        dt: Optional[pd.Timedelta] = None,
    ) -> "VintageTransformer":
        policy = policy or self.policy
        vintage = normalize_vintage_frame(
            vintage_frame,
            availability_margin=policy.availability_margin,
            signal_names=self.names,
        )
        target_dt = pd.Timedelta(dt or self.dt or "15min")
        vintage = densify_vintage_frame(vintage, target_dt)
        self._snapshots = _snapshot_table(vintage)
        self._lookups = {
            name: _SignalLookup(vintage, name, self._snapshots) for name in self.names
        }
        self.policy = policy
        self.dt = target_dt
        return self

    def transform(
        self,
        origins: pd.DatetimeIndex,
        vintage_frame: Optional[pd.DataFrame] = None,
        policy: Optional[VintagePolicy] = None,
        return_provenance: Optional[bool] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        policy = policy or self.policy
        if vintage_frame is not None or not self._lookups:
            self.prepare(vintage_frame, policy=policy)

        origins_in = pd.DatetimeIndex(origins)
        origins_utc = _ensure_utc_index(origins_in)
        if self.dt is None:
            if len(origins_utc) < 2:
                raise ValueError("dt must be set when transforming fewer than 2 origins")
            dt = pd.Series(origins_utc).diff().median()
        else:
            dt = pd.Timedelta(self.dt)
        self._ensure_metadata(dt)

        snap_codes, selected_avail = select_snapshot_codes(
            origins_utc, self._snapshots, mode=policy.mode, fallback=policy.fallback
        )
        if policy.assert_causal and policy.mode == "latest_available":
            assert_causal_selection(origins_utc, selected_avail)

        feature_cols = list(self.generated_features)
        out = np.full((len(origins_utc), len(feature_cols)), np.nan, dtype=np.float32)
        col_index = {c: i for i, c in enumerate(feature_cols)}

        # Fast path: point samples with lags and no aggregation functions.
        use_fast = self.functions is None and self.base.agg_bins is None

        origin_ns = origins_utc.tz_convert("UTC").tz_localize(None).asi8.astype(np.int64)
        dt_ns = int(pd.Timedelta(dt).value)
        chunk = policy.chunk_size

        for name in self.names:
            lookup = self._lookups[name]
            meta_name = self.metadata.loc[self.metadata["name"] == name]

            for feat_name, row in meta_name.iterrows():
                j = col_index[feat_name]
                lag = 0 if pd.isna(row.get("lag")) else int(row["lag"])
                fun = row.get("function")
                if isinstance(fun, float) and np.isnan(fun):
                    fun = "none"
                fun = "none" if fun is None else str(fun)

                if use_fast or fun in {"none", "None"}:
                    # point value at origin - lag * dt, matching Transformer.shift(lag)
                    target_ns = origin_ns - lag * dt_ns
                    for start in range(0, len(origins_utc), chunk):
                        stop = min(start + chunk, len(origins_utc))
                        out[start:stop, j] = lookup.gather(
                            snap_codes[start:stop], target_ns[start:stop]
                        )
                else:
                    start_delta = pd.Timedelta(row["start_time"])
                    end_delta = pd.Timedelta(row["end_time"])
                    start_ns = origin_ns + int(start_delta.value)
                    end_ns = origin_ns + int(end_delta.value)
                    for i in range(len(origins_utc)):
                        out[i, j] = lookup.gather_window(
                            int(snap_codes[i]), start_ns[i], end_ns[i], fun
                        )

        features = pd.DataFrame(out, index=origins_in, columns=feature_cols)
        want_prov = policy.return_provenance if return_provenance is None else return_provenance
        if want_prov:
            snap_times = self._snapshots.set_index("snapshot_code")["snapshot_time"]
            provenance = pd.DataFrame(
                {
                    "snapshot_code": snap_codes,
                    "snapshot_time": [
                        snap_times.get(int(c), pd.NaT) if c >= 0 else pd.NaT for c in snap_codes
                    ],
                    "available_at": selected_avail,
                },
                index=origins_in,
            )
            return features, provenance
        return features


def materialize_vintage_features(
    origins: pd.DatetimeIndex,
    vintage_frame: pd.DataFrame,
    transformers: Sequence[VintageTransformer],
    policy: Optional[VintagePolicy] = None,
    return_provenance: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Apply multiple vintage transformers with shared snapshot preparation."""
    origins = _ensure_utc_index(pd.DatetimeIndex(origins))
    if not transformers:
        empty = pd.DataFrame(index=origins)
        if return_provenance:
            return empty, pd.DataFrame(index=origins)
        return empty

    policy = policy or transformers[0].policy
    dt = None
    for tr in transformers:
        dt = dt or tr.dt
        tr.prepare(vintage_frame, policy=policy, dt=dt)

    frames = []
    provenance = None
    for i, tr in enumerate(transformers):
        result = tr.transform(
            origins,
            vintage_frame=None,
            policy=policy,
            return_provenance=return_provenance and i == 0,
        )
        if return_provenance and i == 0:
            feat, provenance = result
            frames.append(feat)
        else:
            frames.append(result)

    features = pd.concat(frames, axis=1)
    if return_provenance:
        return features, provenance
    return features
