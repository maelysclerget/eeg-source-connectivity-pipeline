"""
Utility functions for the TI/MRCP EEG pipeline.

This module focuses on removing inter-block breaks by:
- locating block start/end triggers (default: S  4 → S  8)
- cropping each block from S4 to (S8 + post_end_sec)
- resampling each cropped block (default: 250 Hz)
- concatenating blocks into a single continuous Raw object
- saving to .fif

Designed for BrainVision recordings (.vhdr/.vmrk/.eeg) using MNE-Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import re
import os
import getpass
import shutil

import numpy as np
import mne

import builtins
import contextlib
import time

SubjectToken = Union[int, str]

class NoBlockMarkersError(RuntimeError):
    """Raised when start/end block markers (e.g., S4/S8) are missing or cannot be paired."""
    def __init__(self, vhdr_name: str, start_code: str, end_code: str, n_start: int, n_end: int):
        self.vhdr_name = vhdr_name
        self.start_code = start_code
        self.end_code = end_code
        self.n_start = n_start
        self.n_end = n_end
        super().__init__(
            f"No valid blocks found in {vhdr_name}. "
            f"Found {n_start} start markers ({start_code}) and {n_end} end markers ({end_code})."
        )

# ---------------------------------------------------------------------
# Portable Hummel root discovery (FROM PREVIOUS CODE)
# ---------------------------------------------------------------------
def _find_hummel_root() -> Path:
    """
    _find_hummel_root allows the code to be run on any operating system, which removes the
    issue of having to change the file path between Mac and Linux

    Returns: File path if it exists, otherwise raises FileNotFoundError
    """
    env = os.environ.get("HUMMEL_DATA", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    candidates = [
        Path("/Volumes/Hummel-Data"),                                  # macOS
        Path("/mnt/Hummel-Data"),                                       # Linux
        Path("/media") / getpass.getuser() / "Hummel-Data",             # Linux (desktop)
        Path("Z:/Hummel-Data"), Path("Y:/Hummel-Data"), Path("X:/Hummel-Data"),  # Windows mapped drives
        Path(r"\\HUMMEL\Hummel-Data"),                                  # Windows UNC (example)
        Path(r"\\SERVER\Hummel-Data"),
        Path("Z:/"),                                                   # another UNC pattern (adjust if needed)
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not locate 'Hummel-Data'. Set HUMMEL_DATA env var to the mounted path."
    )


def default_data_root() -> Path:
    """
    Default EEG base path derived from HUMMEL_ROOT.

    As requested:
      BASE = HUMMEL_ROOT / "TI" / "TI_EEG" / "EEG"
    """
    hummel_root = _find_hummel_root()
    return hummel_root / "TI" / "TI_EEG" / "EEG"


@dataclass(frozen=True)
class BlockWindow:
    """Time window (seconds) in the source Raw to keep for a block."""
    tmin: float
    tmax: float  # inclusive tmax for crop(include_tmax=True)


def normalize_marker_description(desc: str) -> str:
    """
    Normalize annotation/marker descriptions to improve matching robustness.

    Examples:
      "Stimulus/S  4" -> "S 4"
      "S  4"          -> "S 4"
      "S4"            -> "S 4"
    """
    if desc is None:
        return ""
    s = str(desc).strip()
    s = s.replace("Stimulus/", "").replace("Response/", "").replace("Comment/", "")
    s = s.replace("\\1", ",")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^([SR])\s*0*([0-9]+)$", r"\1 \2", s)
    return s


def find_annotation_onsets(raw: mne.io.BaseRaw, code: str) -> np.ndarray:
    """Return sorted onsets (seconds) for annotations matching the given marker code."""
    wanted = normalize_marker_description(code)
    if raw.annotations is None or len(raw.annotations) == 0:
        return np.array([], dtype=float)

    descs = [normalize_marker_description(d) for d in raw.annotations.description]
    onsets = np.asarray(raw.annotations.onset, dtype=float)

    mask = np.array(
        [(d == wanted) or d.endswith(wanted) or wanted.endswith(d) for d in descs],
        dtype=bool,
    )
    return np.sort(onsets[mask])


def pair_block_windows(
    s4_onsets: Sequence[float],
    s8_onsets: Sequence[float],
    post_end_sec: float,
    raw_tmax: float,
    require_s8_before_next_s4: bool = True,
) -> List[BlockWindow]:
    """Pair each S4 onset with the next S8 onset to form block windows."""
    s4 = np.asarray(sorted(s4_onsets), dtype=float)
    s8 = np.asarray(sorted(s8_onsets), dtype=float)
    windows: List[BlockWindow] = []

    if len(s4) == 0:
        return windows

    j = 0
    for i, t4 in enumerate(s4):
        while j < len(s8) and s8[j] <= t4:
            j += 1
        if j >= len(s8):
            continue

        t8 = float(s8[j])

        if require_s8_before_next_s4 and i < len(s4) - 1:
            t4_next = float(s4[i + 1])
            if t8 >= t4_next:
                continue

        tmin = max(0.0, float(t4))
        tmax = min(float(raw_tmax), float(t8 + post_end_sec))
        if tmax > tmin:
            windows.append(BlockWindow(tmin=tmin, tmax=tmax))

    return windows

class UnreadableBrainVisionError(RuntimeError):
    """Raised when a BrainVision file set (.vhdr/.eeg/.vmrk) cannot be opened/read."""
    pass


def _parse_vhdr_sidecars(vhdr_path: Path) -> tuple[str, str]:
    """
    Read DataFile and MarkerFile from the .vhdr header.
    Returns (data_fname, marker_fname) exactly as written in the header.
    """
    data_file = None
    marker_file = None
    for line in vhdr_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("DataFile="):
            data_file = line.split("=", 1)[1].strip()
        elif line.startswith("MarkerFile="):
            marker_file = line.split("=", 1)[1].strip()
        if data_file and marker_file:
            break
    if not data_file or not marker_file:
        raise UnreadableBrainVisionError(f"Could not parse DataFile/MarkerFile from {vhdr_path}")
    return data_file, marker_file


def _cache_brainvision_locally(vhdr_path: Path, cache_root: Path) -> Path:
    """
    Copy the BrainVision triplet (.vhdr/.vmrk/.eeg) into a local cache folder and
    write a patched .vhdr that points to the local .eeg/.vmrk.

    Returns path to the cached .vhdr.
    """
    vhdr_path = vhdr_path.resolve()
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    # Per-file cache folder to avoid collisions
    dst_dir = cache_root / vhdr_path.stem
    dst_dir.mkdir(parents=True, exist_ok=True)

    data_file, marker_file = _parse_vhdr_sidecars(vhdr_path)

    src_eeg = (vhdr_path.parent / data_file).resolve()
    src_vmrk = (vhdr_path.parent / marker_file).resolve()

    if not src_eeg.exists():
        raise UnreadableBrainVisionError(f"Missing .eeg referenced by {vhdr_path.name}: {src_eeg}")
    if not src_vmrk.exists():
        raise UnreadableBrainVisionError(f"Missing .vmrk referenced by {vhdr_path.name}: {src_vmrk}")

    # Copy sidecars + header
    dst_eeg = dst_dir / src_eeg.name
    dst_vmrk = dst_dir / src_vmrk.name
    dst_vhdr = dst_dir / vhdr_path.name

    # Only copy if not already present (saves time when rerunning)
    if not dst_eeg.exists() or dst_eeg.stat().st_size != src_eeg.stat().st_size:
        shutil.copy2(src_eeg, dst_eeg)
    if not dst_vmrk.exists() or dst_vmrk.stat().st_size != src_vmrk.stat().st_size:
        shutil.copy2(src_vmrk, dst_vmrk)

    # Patch vhdr so DataFile/MarkerFile point to local copies
    text = vhdr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    patched = []
    for line in text:
        if line.strip().startswith("DataFile="):
            patched.append(f"DataFile={dst_eeg.name}")
        elif line.strip().startswith("MarkerFile="):
            patched.append(f"MarkerFile={dst_vmrk.name}")
        else:
            patched.append(line)
    dst_vhdr.write_text("\n".join(patched) + "\n", encoding="utf-8")

    return dst_vhdr


import builtins
import contextlib
from pathlib import Path

@contextlib.contextmanager
def _ignore_bad_fd_on_close_for_eeg(eeg_path: Path):
    """
    Monkeypatch builtins.open ONLY for the specific BrainVision .eeg file.
    For all other files, use the original open unchanged.

    This avoids breaking matplotlib/mne imports that read config files via open().
    """
    eeg_path = Path(eeg_path).resolve()
    orig_open = builtins.open

    def patched_open(file, *args, **kwargs):
        # Preserve normal behavior for non-target files
        try:
            fpath = Path(file).resolve()
        except Exception:
            fpath = None

        f = orig_open(file, *args, **kwargs)

        # Only wrap the EEG binary file
        if fpath is None or fpath != eeg_path:
            return f

        class _WrappedFile:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            # Make it iterable to avoid surprises (even though .eeg is binary)
            def __iter__(self):
                return iter(self._inner)

            def close(self):
                try:
                    return self._inner.close()
                except OSError as e:
                    if getattr(e, "errno", None) == 9:
                        return None
                    raise

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                # Ensure close is called; swallow Errno 9 in close()
                self.close()
                return False

        return _WrappedFile(f)

    builtins.open = patched_open
    try:
        yield
    finally:
        builtins.open = orig_open


import time

def read_raw_brainvision_resilient(
    vhdr_path: Path,
    verbose: str = "ERROR",
    retries: int = 3,
    retry_sleep_sec: float = 1.0,
) -> mne.io.BaseRaw:
    """
    Read BrainVision from a flaky network mount by suppressing Errno 9 only when
    closing the specific .eeg file referenced by the .vhdr.

    Avoids global open() monkeypatch side effects.
    """
    vhdr_path = Path(vhdr_path).expanduser().resolve()

    # Parse the .eeg filename from the header so we can target it precisely
    data_file, _marker_file = _parse_vhdr_sidecars(vhdr_path)
    eeg_path = (vhdr_path.parent / data_file).resolve()

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with _ignore_bad_fd_on_close_for_eeg(eeg_path):
                return mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=verbose)
        except OSError as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep_sec * attempt)
                continue
            raise UnreadableBrainVisionError(
                f"Failed to read BrainVision {vhdr_path.name} after {retries} attempt(s). "
                f"Last error: {repr(last_err)}"
            )


def crop_resample_concatenate(
    vhdr_path: Union[str, Path],
    start_code: str = "S  4",
    end_code: str = "S  8",
    post_end_sec: float = 50.0,
    target_sfreq: Optional[float] = 250.0,
    require_s8_before_next_s4: bool = True,
    verbose: Union[bool, str] = "ERROR",
    cache_root: Path | None = None,
) -> Tuple[mne.io.Raw, List[BlockWindow]]:
    """
    Load a BrainVision file, crop each block, resample each cropped block (optional),
    and concatenate blocks.
    """
    vhdr_path = Path(vhdr_path).expanduser().resolve()
    if not vhdr_path.exists():
        raise FileNotFoundError(f"Missing .vhdr file: {vhdr_path}")

    raw = read_raw_brainvision_resilient(Path(vhdr_path), verbose=verbose, retries=3)
    raw_tmax = float(raw.times[-1])

    s4_onsets = find_annotation_onsets(raw, start_code)
    s8_onsets = find_annotation_onsets(raw, end_code)

    windows = pair_block_windows(
        s4_onsets=s4_onsets,
        s8_onsets=s8_onsets,
        post_end_sec=post_end_sec,
        raw_tmax=raw_tmax,
        require_s8_before_next_s4=require_s8_before_next_s4,
    )

    if len(windows) == 0:
        raise NoBlockMarkersError(
            vhdr_name=vhdr_path.name,
            start_code=start_code,
            end_code=end_code,
            n_start=len(s4_onsets),
            n_end=len(s8_onsets),
        )


    cropped_blocks: List[mne.io.Raw] = []
    for w in windows:
        seg = raw.copy().crop(tmin=w.tmin, tmax=w.tmax, include_tmax=True)
        seg.load_data()
        if target_sfreq is not None:
            seg.resample(float(target_sfreq), npad="auto")
        cropped_blocks.append(seg)

    raw_concat = mne.concatenate_raws(cropped_blocks, preload=True, verbose=verbose)
    return raw_concat, windows


def resolve_subject_dir(data_root: Path, subject: SubjectToken) -> Path:
    """
    Resolve a subject token (e.g., 41 or "41Y03") to an existing subject directory.
    """
    data_root = Path(data_root).expanduser().resolve()
    subj_str = str(subject).strip()

    direct = data_root / subj_str
    if direct.exists() and direct.is_dir():
        return direct

    if subj_str.isdigit():
        matches = sorted([p for p in data_root.glob(f"{subj_str}*") if p.is_dir()])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            chosen = matches[0]
            raise RuntimeError(
                f"Ambiguous subject '{subject}': multiple matches under {data_root}: "
                f"{[m.name for m in matches]}. "
                f"Please pass the full subject folder name (e.g., '{chosen.name}')."
            )

    raise FileNotFoundError(f"Could not resolve subject directory for '{subject}' under {data_root}")


def list_available_tasks(subject_dir: Path, session: int) -> List[str]:
    """List tasks (file stems after '<SUBJ>_ses<session>_') available as .vhdr files."""
    subject_dir = Path(subject_dir).expanduser().resolve()
    subj = subject_dir.name
    pat = f"{subj}_ses{session}_*.vhdr"
    tasks: List[str] = []
    for p in sorted(subject_dir.glob(pat)):
        stem = p.stem
        prefix = f"{subj}_ses{session}_"
        if stem.startswith(prefix):
            tasks.append(stem[len(prefix):])
    return tasks


def build_vhdr_path(subject_dir: Path, session: int, task: str) -> Path:
    """Construct expected .vhdr file path for a subject/session/task."""
    subject_dir = Path(subject_dir).expanduser().resolve()
    subj = subject_dir.name
    return subject_dir / f"{subj}_ses{session}_{task}.vhdr"
