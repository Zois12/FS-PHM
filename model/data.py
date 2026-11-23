"""HDF5 data discovery, lazy loading, and leakage-safe PHM splits.

The files under ``splitData`` are not uniform: some store one sensor in
``signal``, some store several named sensors, and RUL trajectories use a
different number of windows per bearing.  This module keeps the HDF5 files
lazy and normalizes every row to a ``(channels, time)`` NumPy array.

Pretraining datasets intentionally omit ``label`` and every known RUL field
from their returned samples.  Few-shot selection is group aware by default;
the selected groups can be excluded from pretraining through the generated
manifest so adjacent windows from the same run cannot leak across stages.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

try:  # h5py is a data dependency, but keeping import errors actionable helps CLI users.
    import h5py
except ImportError as exc:  # pragma: no cover - exercised only in an incomplete environment
    h5py = None  # type: ignore[assignment]
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


FD_DATASETS: Tuple[str, ...] = ("CWRU", "JNU", "PU", "UO")
RUL_DATASETS: Tuple[str, ...] = ("XJTU-SY", "IMS", "PHM2012")

RUL_KEY_ALIASES: Tuple[str, ...] = (
    "rul_norm",
    "rul",
    "rul_min",
    "rul_minutes",
    "rul_sec",
    "rul_seconds",
    "RUL",
)
LABEL_KEY_ALIASES: Tuple[str, ...] = ("label", "labels", "y", "class")

_FD_SIGNAL_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("vibration", "phase_current_1", "phase_current_2"),
    ("DE", "FE"),
    ("signal",),
)
_RUL_SIGNAL_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("signal",),
    ("Horizontal", "Vertical"),
    ("horizontal", "vertical"),
)
RPM_TRACE_ALIASES: Tuple[str, ...] = ("rpm_trace", "instantaneous_rpm", "shaft_speed_rpm")
SHAFT_ANGLE_ALIASES: Tuple[str, ...] = ("shaft_angle_rad", "angle_rad")


def _require_h5py() -> None:
    if h5py is None:  # pragma: no cover - environment setup error
        raise ImportError(
            "h5py is required for PHM HDF5 loading. Install the dependencies "
            "listed in model/requirements.txt."
        ) from _H5PY_IMPORT_ERROR


def _decode(value: Any) -> Any:
    """Decode HDF5 byte strings and NumPy scalar wrappers."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    value = _decode(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normal_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _find_key(keys: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    by_normal = {_normal_key(k): k for k in keys}
    for alias in aliases:
        found = by_normal.get(_normal_key(alias))
        if found is not None:
            return found
    return None


def canonical_dataset_name(path: Path) -> str:
    """Map known H5 roots, including nested class folders, to dataset names."""

    text = "/".join(part.lower() for part in path.parts)
    if "cwru_h5" in text:
        return "CWRU"
    if "jnu_h5" in text:
        return "JNU"
    if "pu_h5" in text:
        return "PU"
    if "uo_h5" in text:
        return "UO"
    if "ims_h5" in text:
        return "IMS"
    if "phm2012_h5" in text:
        return "PHM2012"
    if "xjtu-sy_h5" in text or "xjtu_sy_h5" in text:
        return "XJTU-SY"
    return path.parent.name or "unknown"


def default_dataset_roots(data_root: Path, task: str = "all") -> List[Path]:
    """Return the repository's canonical H5 roots.

    The 0.1 second ``XJTU-SY_h5_01`` directory is preferred whenever it has
    real ``*.h5`` files.  Pass ``--root splitData/XJTU-SY_h5`` explicitly when
    the 0.2 second source is needed for a data-completeness experiment.
    """

    task = task.lower()
    fd = [data_root / name for name in ("CWRU_h5", "JNU_h5", "PU_h5")]
    xjtu_01 = data_root / "XJTU-SY_h5_01"
    xjtu_base = data_root / "XJTU-SY_h5"
    xjtu = xjtu_01 if xjtu_01.is_dir() and any(xjtu_01.rglob("*.h5")) else xjtu_base
    rul = [xjtu, data_root / "ims_h5", data_root / "PHM2012_h5"]
    if task == "fd":
        return fd
    if task == "rul":
        return rul
    if task in {"all", "mixed", "pretrain"}:
        return fd + rul
    raise ValueError(f"Unknown task {task!r}; expected fd, rul, or all")


def discover_h5_files(
    data_root: str | Path,
    task: str = "all",
    roots: Optional[Sequence[str | Path]] = None,
    datasets: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Discover real ``*.h5`` files, ignoring partial-transfer suffixes."""

    base = Path(data_root)
    candidates = [Path(x) for x in roots] if roots else default_dataset_roots(base, task)
    wanted = {x.upper() for x in datasets} if datasets else None
    files: List[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".h5":
            paths = [candidate]
        elif candidate.is_dir():
            paths = sorted(candidate.rglob("*.h5"))
        else:
            continue
        for path in paths:
            if path.name.startswith(".") or path.name.endswith(".h5.downloading"):
                continue
            if wanted and canonical_dataset_name(path).upper() not in wanted:
                continue
            files.append(path.resolve())
    # An explicit root can overlap a default root.  Preserve order while de-duping.
    files = list(dict.fromkeys(files))
    seen_dataset_names: Set[Tuple[str, str]] = set()
    for path in files:
        identity = (canonical_dataset_name(path), path.name)
        if identity in seen_dataset_names and identity[0] == "XJTU-SY":
            warnings.warn(
                f"Both XJTU-SY window variants contain {path.name}; use one root to avoid duplicate trajectories",
                RuntimeWarning,
            )
        seen_dataset_names.add(identity)
    return files


def _stable_path_id(path: Path, base: Path) -> str:
    """Return a path identifier that survives copying the repository/server data."""

    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def _attribute_period(attrs: Mapping[str, Any], task: str, default: Optional[float]) -> float:
    if default is not None:
        return float(default)
    window_sec = _safe_float(attrs.get("window_sec"))
    if window_sec is not None:
        return window_sec
    window = _safe_float(attrs.get("window"))
    sampling_rate = _safe_float(attrs.get("sampling_rate"))
    if window is not None and sampling_rate and sampling_rate > 0:
        return window / sampling_rate
    return 0.2 if task == "fd" else 0.1


def _is_signal_dataset(dataset: Any, n_samples: int) -> bool:
    return bool(getattr(dataset, "ndim", 0) >= 2 and dataset.shape[0] == n_samples)


def _pick_signal_keys(file: Any, task: str, n_samples_hint: Optional[int] = None) -> Tuple[str, ...]:
    keys = list(file.keys())
    groups = _FD_SIGNAL_GROUPS if task == "fd" else _RUL_SIGNAL_GROUPS
    for group in groups:
        actual = tuple(k for k in group if k in file)
        if not actual:
            continue
        if n_samples_hint is None:
            n_samples_hint = int(file[actual[0]].shape[0])
        if all(_is_signal_dataset(file[k], n_samples_hint) for k in actual):
            return actual

    # A permissive fallback makes synthetic/custom H5 files usable without
    # weakening the explicit aliases above.
    excluded = {
        _normal_key(k)
        for k in (
            *RUL_KEY_ALIASES, *LABEL_KEY_ALIASES, *RPM_TRACE_ALIASES,
            *SHAFT_ANGLE_ALIASES, "rpm", "source", "device",
        )
    }
    numeric: List[str] = []
    for key in keys:
        if _normal_key(key) in excluded:
            continue
        dataset = file[key]
        if n_samples_hint is None and getattr(dataset, "ndim", 0) >= 1:
            n_samples_hint = int(dataset.shape[0])
        if n_samples_hint is not None and _is_signal_dataset(dataset, n_samples_hint):
            numeric.append(key)
    if numeric:
        return tuple(numeric)
    raise ValueError(f"Could not find a signal dataset in H5 keys: {keys}")


@dataclass(frozen=True)
class H5FileInfo:
    path: str
    dataset: str
    task: str
    signal_keys: Tuple[str, ...]
    n_samples: int
    signal_length: int
    n_channels: int
    label_key: Optional[str]
    target_key: Optional[str]
    sample_period: float
    group_fields: Tuple[str, ...]
    rpm_key: Optional[str] = None
    rpm_trace_key: Optional[str] = None
    shaft_angle_key: Optional[str] = None
    nominal_rpm: Optional[float] = None


@dataclass(frozen=True)
class SampleRef:
    path: str
    row: int
    dataset: str
    task: str
    group_id: str
    label: Optional[int] = None
    target: Optional[float] = None
    sample_period: float = 0.1

    @property
    def key(self) -> str:
        return f"{self.path}::{self.row}"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class H5Index:
    """Metadata index over H5 rows; signal values stay on disk until read."""

    def __init__(self, records: Sequence[SampleRef], files: Mapping[str, H5FileInfo]):
        self.records = list(records)
        self.files = dict(files)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[SampleRef]:
        return iter(self.records)

    @classmethod
    def build(
        cls,
        data_root: str | Path = "splitData",
        task: str = "all",
        roots: Optional[Sequence[str | Path]] = None,
        datasets: Optional[Sequence[str]] = None,
        target_key: str = "rul_norm",
        sample_period: Optional[float] = None,
        exclude_paths: Optional[Sequence[str | Path]] = None,
        exclude_groups: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
        strict: bool = True,
        include_supervision: bool = True,
        show_progress: bool = False,
    ) -> "H5Index":
        _require_h5py()
        root_base = Path(data_root).resolve()
        paths = discover_h5_files(data_root, task, roots=roots, datasets=datasets)
        excluded_paths = {str(Path(p).resolve()) for p in (exclude_paths or [])}
        excluded_groups = set(exclude_groups or ())
        excluded_datasets = {str(name).upper() for name in (exclude_datasets or ())}
        records: List[SampleRef] = []
        files: Dict[str, H5FileInfo] = {}

        for path in tqdm(paths, desc="Index H5", unit="file", dynamic_ncols=True, disable=not show_progress):
            if str(path) in excluded_paths:
                continue
            if canonical_dataset_name(path).upper() in excluded_datasets:
                continue
            try:
                with h5py.File(path, "r") as handle:
                    keys = list(handle.keys())
                    label_key = _find_key(keys, LABEL_KEY_ALIASES)
                    found_target = _find_key(keys, RUL_KEY_ALIASES)
                    detected_task = "rul" if found_target else ("fd" if label_key else "")
                    requested_task = task.lower()
                    if requested_task == "pretrain":
                        requested_task = "all"
                    if requested_task in {"fd", "rul"} and detected_task != requested_task:
                        continue
                    if not detected_task:
                        if strict:
                            raise ValueError(f"Cannot infer task for {path}")
                        warnings.warn(f"Skipping task-ambiguous H5 file {path}")
                        continue
                    n_hint: Optional[int] = None
                    if label_key:
                        n_hint = int(handle[label_key].shape[0]) if handle[label_key].ndim else None
                    elif found_target:
                        n_hint = int(handle[found_target].shape[0]) if handle[found_target].ndim else None
                    signal_keys = _pick_signal_keys(handle, detected_task, n_hint)
                    n_samples = int(handle[signal_keys[0]].shape[0])
                    if any(int(handle[k].shape[0]) != n_samples for k in signal_keys):
                        raise ValueError(f"Signal row counts disagree in {path}: {signal_keys}")

                    target_actual = None
                    if detected_task == "rul":
                        requested = _find_key(keys, (target_key,)) if target_key else None
                        if include_supervision and target_key and _normal_key(target_key) not in {"rul_norm", "auto"} and requested is None:
                            raise ValueError(f"Requested RUL target {target_key!r} is absent from {path}")
                        target_actual = requested or found_target
                        if target_actual is None:
                            raise ValueError(f"No RUL target key in {path}")
                        if handle[target_actual].ndim and int(handle[target_actual].shape[0]) != n_samples:
                            raise ValueError(f"Target row count disagrees with signal in {path}")
                    if detected_task == "fd" and label_key is not None:
                        if handle[label_key].ndim and int(handle[label_key].shape[0]) != n_samples:
                            raise ValueError(f"Label row count disagrees with signal in {path}")

                    first_shape = handle[signal_keys[0]].shape
                    if len(first_shape) == 2:
                        n_channels, signal_length = 1, int(first_shape[1])
                    else:
                        n_channels, signal_length = int(first_shape[1]), int(np.prod(first_shape[2:]))
                    if len(signal_keys) > 1:
                        n_channels *= len(signal_keys)

                    attrs = {str(k): _decode(v) for k, v in handle.attrs.items()}
                    period = _attribute_period(attrs, detected_task, sample_period)
                    rpm_key = _find_key(keys, ("rpm",))
                    rpm_trace_key = _find_key(keys, RPM_TRACE_ALIASES)
                    scalar_rpm = None
                    if rpm_key is not None:
                        if handle[rpm_key].ndim == 0:
                            scalar_rpm = _safe_float(handle[rpm_key][()])
                            rpm_key = None
                        elif handle[rpm_key].ndim >= 2 and int(np.prod(handle[rpm_key].shape[1:])) > 1:
                            if rpm_trace_key is None:
                                rpm_trace_key = rpm_key
                            rpm_key = None
                    shaft_angle_key = _find_key(keys, SHAFT_ANGLE_ALIASES)
                    for speed_key in (rpm_key, rpm_trace_key, shaft_angle_key):
                        if speed_key is not None and int(handle[speed_key].shape[0]) != n_samples:
                            raise ValueError(f"Speed metadata row count disagrees with signal in {path}")
                    nominal_rpm = _safe_float(
                        attrs.get("speed_rpm"), _safe_float(attrs.get("rpm"), scalar_rpm)
                    )
                    group_fields = tuple(k for k in ("source", "device") if k in handle)
                    info = H5FileInfo(
                        path=str(path),
                        dataset=canonical_dataset_name(path),
                        task=detected_task,
                        signal_keys=signal_keys,
                        n_samples=n_samples,
                        signal_length=signal_length,
                        n_channels=n_channels,
                        label_key=label_key,
                        target_key=target_actual,
                        sample_period=period,
                        group_fields=group_fields,
                        rpm_key=rpm_key,
                        rpm_trace_key=rpm_trace_key,
                        shaft_angle_key=shaft_angle_key,
                        nominal_rpm=nominal_rpm,
                    )
                    files[str(path)] = info

                    labels = _read_row_values(handle, label_key, n_samples, default=None) if include_supervision else None
                    targets = _read_row_values(handle, target_actual, n_samples, default=None) if include_supervision else None
                    if include_supervision and detected_task == "rul" and target_key and _normal_key(target_key) == "rul_norm" and target_actual:
                        # Some custom files have only a raw RUL column.  Keep the
                        # canonical normalized target semantics in that case.
                        if _normal_key(target_actual) != "rul_norm" and targets is not None:
                            arr = np.asarray(targets, dtype=np.float64)
                            finite = arr[np.isfinite(arr)]
                            scale = float(np.max(np.abs(finite))) if finite.size else 1.0
                            if scale > 1.0:
                                targets = (arr / scale).tolist()

                    # Consolidated CWRU files carry source/device columns, while
                    # per-run H5 files omit them.  Keep both layouts lazy and
                    # treat these metadata fields as optional.
                    source_key = "source" if "source" in handle else None
                    device_key = "device" if "device" in handle else None
                    source_values = _read_row_values(handle, source_key, n_samples, default=None)
                    device_values = _read_row_values(handle, device_key, n_samples, default=None)
                    relative_file = _stable_path_id(path, root_base)
                    file_group = f"{info.dataset}::{relative_file}"
                    for row in range(n_samples):
                        if source_values is not None or device_values is not None:
                            source = _decode(source_values[row]) if source_values is not None else ""
                            device = _decode(device_values[row]) if device_values is not None else ""
                            group_id = (
                                f"{info.dataset}::{relative_file}::source={source}::device={device}"
                            )
                        else:
                            group_id = file_group
                        if group_id in excluded_groups:
                            continue
                        label = _coerce_label(labels[row]) if labels is not None else None
                        target = _coerce_target(targets[row]) if targets is not None else None
                        records.append(
                            SampleRef(
                                path=str(path),
                                row=row,
                                dataset=info.dataset,
                                task=detected_task,
                                group_id=group_id,
                                label=label,
                                target=target,
                                sample_period=period,
                            )
                        )
            except Exception:
                if strict:
                    raise
                warnings.warn(f"Skipping unreadable H5 file {path}", RuntimeWarning)

        if not records:
            warnings.warn("H5 index is empty; check task, roots, and exclusion arguments")
        return cls(records, files)

    def filter(
        self,
        task: Optional[str] = None,
        datasets: Optional[Sequence[str]] = None,
        groups: Optional[Iterable[str]] = None,
        paths: Optional[Iterable[str | Path]] = None,
    ) -> "H5Index":
        wanted_task = task.lower() if task else None
        wanted_datasets = {x.upper() for x in datasets} if datasets else None
        wanted_groups = set(groups) if groups is not None else None
        wanted_paths = {str(Path(p).resolve()) for p in paths} if paths is not None else None
        selected = [
            ref
            for ref in self.records
            if (wanted_task is None or ref.task == wanted_task)
            and (wanted_datasets is None or ref.dataset.upper() in wanted_datasets)
            and (wanted_groups is None or ref.group_id in wanted_groups)
            and (wanted_paths is None or str(Path(ref.path).resolve()) in wanted_paths)
        ]
        return H5Index(selected, self.files)

    def by_key(self) -> Dict[str, SampleRef]:
        return {ref.key: ref for ref in self.records}


def _read_row_values(handle: Any, key: Optional[str], n_samples: int, default: Any) -> Optional[List[Any]]:
    if key is None or key not in handle:
        return None
    dataset = handle[key]
    if dataset.ndim == 0:
        value = _decode(dataset[()])
        return [value] * n_samples
    values = dataset[:]
    return [_decode(v) for v in values]


def _coerce_label(value: Any) -> Optional[int]:
    try:
        number = int(_decode(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _coerce_target(value: Any) -> Optional[float]:
    result = _safe_float(value)
    return result


def _row_to_channels(value: np.ndarray) -> List[np.ndarray]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [arr.reshape(1)]
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        # H5 signal rows conventionally use (channels, time).  The fallback
        # handles (time, channels) custom files without transposing long waves.
        if arr.shape[0] <= 32 and arr.shape[1] >= arr.shape[0]:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.shape[1] <= 32 and arr.shape[0] > arr.shape[1]:
            return [arr[:, i] for i in range(arr.shape[1])]
        return [arr.reshape(-1)]
    return [part.reshape(-1) for part in arr.reshape(arr.shape[0], -1)]


def _normalize_channels(signal: np.ndarray, length: int, eps: float = 1e-6) -> np.ndarray:
    result = np.asarray(signal, dtype=np.float32).copy()
    valid = result[:, :length]
    finite = np.isfinite(valid)
    result[~np.isfinite(result)] = 0.0
    if length <= 0:
        return result
    for channel in range(result.shape[0]):
        values = valid[channel][finite[channel]]
        if values.size == 0:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale < eps:
            scale = float(np.std(values))
        if scale < eps:
            scale = 1.0
        result[channel, :length] = (result[channel, :length] - median) / scale
    return result


class PHMH5Dataset:
    """Lazy row-level dataset for a previously built :class:`H5Index`.

    The object is deliberately framework neutral and returns NumPy arrays.
    Pretraining also returns the next window from a retained RUL trajectory
    for label-free monotonicity training.  A PyTorch ``collate_batch`` is
    provided below and imports torch only when it is actually called.
    """

    def __init__(
        self,
        index: H5Index,
        mode: str = "finetune",
        robust_normalize: bool = True,
        include_metadata: bool = True,
    ) -> None:
        mode = mode.lower()
        if mode not in {"pretrain", "finetune", "eval"}:
            raise ValueError("mode must be pretrain, finetune, or eval")
        self.index = index
        self.records = index.records
        self.mode = mode
        self.robust_normalize = robust_normalize
        self.include_metadata = include_metadata
        self.power_spectrum = None
        self._handles: Dict[str, Any] = {}
        self._pair_valid = np.zeros(len(self.records), dtype=np.bool_)
        if mode == "pretrain":
            # A pair is usable only when both adjacent windows survived index
            # exclusions and belong to the same trajectory group.
            positions: Dict[str, np.ndarray] = {}
            for path, info in index.files.items():
                if info.task == "rul":
                    positions[path] = np.full(info.n_samples, -1, dtype=np.int64)
            for position, ref in enumerate(self.records):
                if ref.task == "rul":
                    positions[ref.path][ref.row] = position
            for position, ref in enumerate(self.records):
                if ref.task != "rul" or ref.row + 1 >= len(positions[ref.path]):
                    continue
                next_position = positions[ref.path][ref.row + 1]
                if next_position >= 0 and self.records[next_position].group_id == ref.group_id:
                    self._pair_valid[position] = True

    def __len__(self) -> int:
        return len(self.records)

    def _handle(self, path: str) -> Any:
        _require_h5py()
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def _load_signal(self, info: H5FileInfo, row: int) -> Tuple[np.ndarray, int]:
        handle = self._handle(info.path)
        channels: List[np.ndarray] = []
        for key in info.signal_keys:
            channels.extend(_row_to_channels(np.asarray(handle[key][row])))
        if not channels:
            raise RuntimeError(f"No signal channels found for {info.path} row {row}")
        length = max(int(channel.shape[-1]) for channel in channels)
        signal = np.zeros((len(channels), length), dtype=np.float32)
        for channel_index, channel in enumerate(channels):
            count = min(length, int(channel.shape[-1]))
            signal[channel_index, :count] = np.asarray(channel[:count], dtype=np.float32)
        if self.robust_normalize:
            signal = _normalize_channels(signal, length)
        else:
            signal[~np.isfinite(signal)] = 0.0
        return signal, length

    def _power_channels(self, ref):
        info = self.index.files[ref.path]
        handle = self._handle(ref.path)
        channels = []
        for key in info.signal_keys:
            rows = _row_to_channels(np.asarray(handle[key][ref.row]))
            channels.extend((key if len(rows) == 1 else f"{key}:{i}", row) for i, row in enumerate(rows))
        return sorted(channels, key=lambda x: x[0])

    def configure_power_spectrum(self, bins=256, max_hz=6000., channel_names=None):
        """Enable raw, per-channel descriptors, before normalization or padding.

        Channel order is fixed by the training schema and must match at eval.
        No supervised fitting or query-derived normalization happens here.
        """
        if not self.records or any(r.task != "fd" for r in self.records):
            raise ValueError("Power spectra require a nonempty FD dataset")
        names = [name for name, _ in self._power_channels(self.records[0])]
        if len(names) != len(set(names)):
            raise ValueError("Ambiguous power-spectrum channel names")
        if channel_names is not None and names != list(channel_names):
            raise ValueError("Power-spectrum channel schema differs from training")
        self.power_spectrum = {"bins": bins, "max_hz": max_hz, "channel_names": names}
        return names

    def _power_features(self, ref):
        from .signal_spectrum import spectral_features

        channels = self._power_channels(ref)
        cfg = self.power_spectrum
        if [name for name, _ in channels] != cfg["channel_names"]:
            raise ValueError(f"Power-spectrum channel schema changed in {ref.path}")
        return np.concatenate([spectral_features(row, len(row) / ref.sample_period, cfg["bins"], cfg["max_hz"])
                               for _, row in channels]).astype(np.float32)

    def _rotation_metadata(self, info: H5FileInfo, row: int, length: int) -> Dict[str, Any]:
        handle = self._handle(info.path)
        metadata: Dict[str, Any] = {}
        if info.rpm_trace_key is not None:
            trace = np.asarray(handle[info.rpm_trace_key][row], dtype=np.float32).reshape(-1)
            if trace.size != length:
                raise ValueError(f"RPM trace length differs from signal in {info.path}, row {row}")
            metadata["rpm_trace"] = trace
        if info.shaft_angle_key is not None:
            angle = np.asarray(handle[info.shaft_angle_key][row], dtype=np.float32).reshape(-1)
            if angle.size != length:
                raise ValueError(f"Shaft angle length differs from signal in {info.path}, row {row}")
            metadata["shaft_angle"] = angle
        rpm = (
            _safe_float(np.asarray(handle[info.rpm_key][row]).reshape(-1)[0])
            if info.rpm_key is not None else info.nominal_rpm
        )
        if rpm is None:
            rpm = info.nominal_rpm
        if rpm is not None and rpm > 0:
            metadata["nominal_rpm"] = np.float32(rpm)
        return metadata

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ref = self.records[index]
        info = self.index.files[ref.path]
        signal, length = self._load_signal(info, ref.row)
        sample: Dict[str, Any] = {"signal": signal, "length": length}
        if self.power_spectrum is not None:
            sample["power_features"] = self._power_features(ref)
        sample.update(self._rotation_metadata(info, ref.row, length))
        if self.mode == "pretrain":
            pair_valid = bool(self._pair_valid[index])
            if pair_valid:
                next_signal, next_length = self._load_signal(info, ref.row + 1)
                next_rotation = self._rotation_metadata(info, ref.row + 1, next_length)
            else:
                next_signal, next_length = np.zeros((1, 1), dtype=np.float32), 1
                next_rotation = {}
            sample.update(
                {
                    "next_signal": next_signal,
                    "next_length": next_length,
                    "pair_valid": pair_valid,
                    **{f"next_{key}": value for key, value in next_rotation.items()},
                }
            )
        else:
            if ref.task == "fd":
                sample["label"] = -1 if ref.label is None else ref.label
            elif ref.task == "rul":
                sample["target"] = np.float32(np.nan if ref.target is None else ref.target)
        if self.include_metadata:
            sample.update(
                {
                    "path": ref.path,
                    "row": ref.row,
                    "dataset": ref.dataset,
                    "task": ref.task,
                    "group_id": ref.group_id,
                    "sample_period": np.float32(ref.sample_period),
                }
            )
        return sample

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies
        self.close()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


def collate_batch(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pad variable channel and time dimensions into a PyTorch batch."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only missing on local setup
        raise ImportError("PyTorch is required for collate_batch and training") from exc

    max_channels = max(int(np.asarray(sample["signal"]).shape[0]) for sample in samples)
    max_length = max(int(sample.get("length", np.asarray(sample["signal"]).shape[-1])) for sample in samples)
    signals = torch.zeros((len(samples), max_channels, max_length), dtype=torch.float32)
    channel_mask = torch.zeros((len(samples), max_channels), dtype=torch.bool)
    lengths = torch.zeros((len(samples),), dtype=torch.long)
    periods = torch.zeros((len(samples),), dtype=torch.float32)
    for i, sample in enumerate(samples):
        array = np.asarray(sample["signal"], dtype=np.float32)
        channels, length = array.shape[0], min(array.shape[-1], max_length)
        signals[i, :channels, :length] = torch.from_numpy(np.ascontiguousarray(array[:, :length]))
        channel_mask[i, :channels] = True
        lengths[i] = int(sample.get("length", length))
        periods[i] = float(sample.get("sample_period", 0.1))

    def rotation_fields(prefix: str, padded_time: int) -> Dict[str, Tensor]:
        traces = torch.zeros((len(samples), padded_time), dtype=torch.float32)
        angles = torch.zeros_like(traces)
        trace_valid = torch.zeros((len(samples),), dtype=torch.bool)
        angle_valid = torch.zeros_like(trace_valid)
        rpm = torch.full((len(samples),), float("nan"), dtype=torch.float32)
        for row, item in enumerate(samples):
            trace = item.get(f"{prefix}rpm_trace")
            if trace is not None:
                values = np.asarray(trace, dtype=np.float32).reshape(-1)
                traces[row, : values.size] = torch.from_numpy(np.ascontiguousarray(values))
                trace_valid[row] = True
            angle = item.get(f"{prefix}shaft_angle")
            if angle is not None:
                values = np.asarray(angle, dtype=np.float32).reshape(-1)
                angles[row, : values.size] = torch.from_numpy(np.ascontiguousarray(values))
                angle_valid[row] = True
            if f"{prefix}nominal_rpm" in item:
                rpm[row] = float(item[f"{prefix}nominal_rpm"])
        return {
            f"{prefix}rpm_traces": traces,
            f"{prefix}rpm_trace_valid": trace_valid,
            f"{prefix}shaft_angles": angles,
            f"{prefix}shaft_angle_valid": angle_valid,
            f"{prefix}nominal_rpm": rpm,
        }

    batch: Dict[str, Any] = {
        "signals": signals,
        "signal": signals,  # compatibility alias for small custom loops
        "lengths": lengths.clamp_min(1).clamp_max(max_length),
        "channel_mask": channel_mask,
        "sample_periods": periods,
        "metadata": [
            {
                key: value
                for key, value in sample.items()
                if key not in {
                    "signal", "length", "label", "target", "next_signal", "next_length", "pair_valid", "power_features",
                    "rpm_trace", "shaft_angle", "nominal_rpm", "next_rpm_trace",
                    "next_shaft_angle", "next_nominal_rpm",
                }
            }
            for sample in samples
        ],
    }
    if any("power_features" in sample for sample in samples):
        if not all("power_features" in sample for sample in samples):
            raise ValueError("Mixed power-spectrum feature configuration within a batch")
        batch["power_features"] = torch.from_numpy(np.stack([sample["power_features"] for sample in samples]))
    batch.update(rotation_fields("", max_length))
    if all("label" in sample for sample in samples):
        batch["labels"] = torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.long)
    if all("target" in sample for sample in samples):
        batch["targets"] = torch.tensor([float(sample["target"]) for sample in samples], dtype=torch.float32)
    if any("next_signal" in sample for sample in samples):
        next_channels = max(
            int(np.asarray(sample.get("next_signal", sample["signal"])).shape[0]) for sample in samples
        )
        next_max_length = max(
            int(sample.get("next_length", sample.get("length", np.asarray(sample["signal"]).shape[-1])))
            for sample in samples
        )
        next_signals = torch.zeros((len(samples), next_channels, next_max_length), dtype=torch.float32)
        next_channel_mask = torch.zeros((len(samples), next_channels), dtype=torch.bool)
        next_lengths = torch.zeros((len(samples),), dtype=torch.long)
        pair_valid = torch.zeros((len(samples),), dtype=torch.bool)
        for i, sample in enumerate(samples):
            if "next_signal" not in sample:
                continue
            array = np.asarray(sample["next_signal"], dtype=np.float32)
            channels, length = array.shape[0], min(array.shape[-1], next_max_length)
            next_signals[i, :channels, :length] = torch.from_numpy(
                np.ascontiguousarray(array[:, :length])
            )
            next_channel_mask[i, :channels] = True
            next_lengths[i] = int(sample.get("next_length", length))
            pair_valid[i] = bool(sample.get("pair_valid", False))
        batch.update(
            {
                "next_signals": next_signals,
                "next_lengths": next_lengths.clamp_min(1).clamp_max(next_max_length),
                "next_channel_mask": next_channel_mask,
                "pair_valid": pair_valid,
            }
        )
        batch.update(rotation_fields("next_", next_max_length))
    return batch


@dataclass
class FewShotSelection:
    task: str
    shots: int
    seed: int
    unit: str
    selected: List[SampleRef]
    excluded_group_ids: List[str]
    split_id: Optional[str] = None
    train_fraction: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "task": self.task,
            "shots": self.shots,
            "seed": self.seed,
            "unit": self.unit,
            "selected": [ref.as_dict() for ref in self.selected],
            "excluded_group_ids": list(self.excluded_group_ids),
            "excluded_datasets": sorted({ref.dataset for ref in self.selected}),
            "split_id": self.split_id,
            "train_fraction": self.train_fraction,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FewShotSelection":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        selected = [SampleRef(**item) for item in payload.get("selected", [])]
        return cls(
            task=str(payload["task"]),
            shots=int(payload["shots"]),
            seed=int(payload.get("seed", 0)),
            unit=str(payload.get("unit", "window")),
            selected=selected,
            excluded_group_ids=list(payload.get("excluded_group_ids", [])),
            split_id=payload.get("split_id"),
            train_fraction=payload.get("train_fraction"),
        )


def select_few_shot(
    records: Iterable[SampleRef] | H5Index,
    shots: int,
    task: str,
    seed: int = 0,
    unit: str = "window",
) -> FewShotSelection:
    """Select deterministic support examples and return groups to hold out.

    FD selects ``shots`` examples per class from distinct groups.  For CWRU a
    group is a source/device run; for JNU/PU it is the H5 file.  RUL selects
    ``shots`` bearing trajectories (one support window per trajectory for
    ``unit='window'`` or all windows for ``unit='group'``).  In both cases the
    selected groups are the exclusion boundary for pretraining.
    """

    if shots < 1:
        raise ValueError("shots must be >= 1")
    task = task.lower()
    if task not in {"fd", "rul"}:
        raise ValueError("few-shot selection requires task='fd' or task='rul'")
    unit = unit.lower()
    if unit not in {"window", "group", "trajectory", "file"}:
        raise ValueError("unit must be window, group, trajectory, or file")
    all_records = list(records.records if isinstance(records, H5Index) else records)
    all_records = [ref for ref in all_records if ref.task == task]
    if not all_records:
        raise ValueError(f"No {task} records available for few-shot selection")
    rng = random.Random(seed)
    selected: List[SampleRef] = []
    excluded_groups: Set[str] = set()

    if task == "fd":
        # Labels are local to a dataset (CWRU label 1 is not JNU label 1), so
        # the selection key is explicitly (dataset, label).
        by_label: Dict[Tuple[str, int], Dict[str, List[SampleRef]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for ref in all_records:
            if ref.label is not None:
                by_label[(ref.dataset, int(ref.label))][ref.group_id].append(ref)
        for dataset_label in sorted(by_label):
            dataset, label = dataset_label
            groups = list(by_label[dataset_label])
            rng.shuffle(groups)
            if unit in {"group", "trajectory", "file"}:
                if len(groups) < shots:
                    raise ValueError(
                        f"{dataset} class {label} has only {len(groups)} independent groups; "
                        f"cannot select {shots} group-shots"
                    )
                for group in groups[:shots]:
                    excluded_groups.add(group)
                    selected.extend(by_label[dataset_label][group])
            else:
                # Prefer one independent run per shot.  If a class has fewer
                # runs than K (common for JNU/CWRU), fill the remaining slots
                # with different windows from already selected runs while
                # still excluding each selected run from pretraining.
                chosen_groups = groups[: min(shots, len(groups))]
                class_selected: List[SampleRef] = []
                pool: List[SampleRef] = []
                used_keys: Set[str] = set()
                for group in chosen_groups:
                    excluded_groups.add(group)
                    candidate = list(by_label[dataset_label][group])
                    rng.shuffle(candidate)
                    if candidate:
                        class_selected.append(candidate[0])
                        used_keys.add(candidate[0].key)
                        pool.extend(candidate[1:])
                if len(class_selected) < shots:
                    for group in groups[min(shots, len(groups)) :]:
                        excluded_groups.add(group)
                        extra = list(by_label[dataset_label][group])
                        rng.shuffle(extra)
                        pool.extend(extra)
                rng.shuffle(pool)
                class_selected.extend(ref for ref in pool if ref.key not in used_keys)
                if len(class_selected) < shots:
                    raise ValueError(
                        f"{dataset} class {label} has only {len(class_selected)} windows; "
                        f"cannot select {shots}-shot"
                    )
                selected.extend(class_selected[:shots])
    else:
        by_group: Dict[str, List[SampleRef]] = defaultdict(list)
        for ref in all_records:
            by_group[ref.group_id].append(ref)
        groups = list(by_group)
        rng.shuffle(groups)
        if unit in {"group", "trajectory", "file"}:
            if len(groups) < shots:
                raise ValueError(f"Only {len(groups)} RUL trajectories available; cannot select {shots} groups")
            for group in groups[:shots]:
                excluded_groups.add(group)
                selected.extend(by_group[group])
        else:
            if not groups:
                raise ValueError("No RUL trajectories available")
            chosen_groups = groups[: min(shots, len(groups))]
            pool: List[SampleRef] = []
            used_keys: Set[str] = set()
            for group in chosen_groups:
                excluded_groups.add(group)
                candidate = list(by_group[group])
                rng.shuffle(candidate)
                if candidate:
                    selected.append(candidate[0])
                    used_keys.add(candidate[0].key)
                    pool.extend(candidate[1:])
            if len(chosen_groups) < shots:
                for group in groups[min(shots, len(groups)) :]:
                    extra = list(by_group[group])
                    rng.shuffle(extra)
                    pool.extend(extra)
            rng.shuffle(pool)
            selected.extend(ref for ref in pool if ref.key not in used_keys)
            # RUL K-shot is defined as K support windows/trajectories.  Keep
            # selected rows deterministic and exclude every source group used.
            selected = selected[:shots]
            excluded_groups = {ref.group_id for ref in selected}

    if unit == "window":
        # Exclude exactly the groups that contribute support rows.  This keeps
        # the manifest tight even when a class needed fallback windows because
        # it had fewer independent runs than the requested K.
        excluded_groups = {ref.group_id for ref in selected}
    selected.sort(key=lambda ref: (ref.dataset, ref.path, ref.row))
    return FewShotSelection(task, int(shots), int(seed), unit, selected, sorted(excluded_groups))


def records_from_selection(index: H5Index, selection: FewShotSelection) -> H5Index:
    """Resolve a manifest against a freshly built index."""

    available = index.by_key()
    missing: List[str] = []
    selected: List[SampleRef] = []
    for ref in selection.selected:
        current = available.get(ref.key)
        if current is None:
            # Manifest paths may point to a different server mount.  Stable
            # group IDs plus dataset/filename/row resolve those manifests
            # without reintroducing ambiguous cross-dataset matches.
            filename = PureWindowsPath(ref.path).name
            candidates = [
                candidate
                for candidate in index.records
                if candidate.dataset == ref.dataset
                and candidate.row == ref.row
                and Path(candidate.path).name == filename
                and candidate.group_id == ref.group_id
            ]
            if len(candidates) == 1:
                current = candidates[0]
        if current is None:
            missing.append(ref.key)
        else:
            selected.append(current)
    if missing:
        raise ValueError(f"Few-shot manifest contains {len(missing)} records missing from the current index")
    return H5Index(selected, index.files)


def write_index_summary(index: H5Index, path: str | Path) -> None:
    """Write a compact, human-readable metadata summary for server jobs."""

    counts: Dict[str, int] = defaultdict(int)
    labels: Dict[str, Set[int]] = defaultdict(set)
    groups: Dict[str, Set[str]] = defaultdict(set)
    for ref in index.records:
        counts[f"{ref.task}:{ref.dataset}"] += 1
        groups[f"{ref.task}:{ref.dataset}"].add(ref.group_id)
        if ref.label is not None:
            labels[f"{ref.task}:{ref.dataset}"].add(ref.label)
    payload = {
        "records": len(index),
        "files": len(index.files),
        "splits": {
            key: {"records": counts[key], "groups": len(groups[key]), "labels": sorted(labels[key])}
            for key in sorted(counts)
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class FewShotSampler:
    """Small stateful wrapper convenient for notebooks and experiment scripts."""

    def __init__(self, shots: int, task: str, seed: int = 0, unit: str = "window") -> None:
        self.shots = int(shots)
        self.task = task
        self.seed = int(seed)
        self.unit = unit

    def select(self, records: Iterable[SampleRef] | H5Index) -> FewShotSelection:
        return select_few_shot(records, self.shots, self.task, seed=self.seed, unit=self.unit)


# Friendly aliases for callers that prefer shorter names.
H5Dataset = PHMH5Dataset
collate_fn = collate_batch

