#!/usr/bin/env python3
"""
remove_breaks_and_save.py

- Find blocks using S4 (start) and S8 (end) triggers
- Keep data from S4 to S8 + 50 seconds
- Resample each cropped block to 250 Hz
- Concatenate all blocks into one continuous Raw
- Save as .fif

If --tasks is omitted, you will be prompted (no task is assumed).

OUTPUT:
- <cwd>/Removing_inter/<participant>/...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import mne

from utils_removal import (
    _find_hummel_root,        # <-- NEW
    build_vhdr_path,
    crop_resample_concatenate,
    list_available_tasks,
    resolve_subject_dir,
    NoBlockMarkersError,   # <-- NEW
)

SubjectToken = Union[int, str]

# ------------------------------------------------------------------
# NEW: portable base path using your HUMMEL_ROOT logic
# ------------------------------------------------------------------
HUMMEL_ROOT = _find_hummel_root()
BASE = HUMMEL_ROOT / "TI" / "TI_EEG" / "Data" / "EEG"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove inter-block breaks and save concatenated .fif")

    p.add_argument("--data-root", help="EEG data root (overrides HUMMEL_ROOT-derived BASE).")
    p.add_argument("--subjects", help="Subject(s): 41 or 41,42 or folder name (e.g., 41Y03). Comma-separated.")
    p.add_argument("--sessions", help="Session(s): 1,2,3 or 'all'.")

    p.add_argument("--tasks", help="Task(s) to process (comma-separated). If omitted, you will be prompted.")

    # Trigger logic
    p.add_argument("--start-code", default="S  4", help="Block start trigger code (default: 'S  4').")
    p.add_argument("--end-code", default="S  8", help="Block end trigger code (default: 'S  8').")
    p.add_argument("--post-end-sec", type=float, default=50.0, help="Seconds to keep after end trigger.")
    p.add_argument("--target-sfreq", type=float, default=250.0, help="Resample each cropped block to this sfreq (Hz).")
    p.add_argument(
        "--allow-s8-after-next-s4",
        action="store_true",
        help="If set, do NOT require S8 to be before the next S4 (less strict pairing).",
    )

    # Output (current working directory / Removing_inter / participant)
    p.add_argument(
        "--out-folder",
        default="Removing_inter",
        help="Folder created under current working directory to store outputs (default: Removing_inter).",
    )
    p.add_argument("--suffix", default="nobreaks", help="Suffix for output .fif filename.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    p.add_argument("--dry-run", action="store_true", help="Only print detected blocks; do not save.")
    
    

    return p.parse_args()


def _parse_csv_tokens(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _parse_subjects(s: Optional[str]) -> List[SubjectToken]:
    toks = _parse_csv_tokens(s)
    out: List[SubjectToken] = []
    for t in toks:
        out.append(int(t) if t.isdigit() else t)
    return out


def _parse_sessions(s: Optional[str]) -> List[int]:
    if not s:
        s = input("Enter session(s) (1,2,3 or 'all'): ").strip().lower()
    s = s.strip().lower()
    return [1, 2, 3] if s == "all" else [int(x.strip()) for x in s.split(",") if x.strip()]


def _choose_tasks_interactive(subject_dir: Path, session: int) -> List[str]:
    available = list_available_tasks(subject_dir, session=session)
    if not available:
        raise FileNotFoundError(
            f"No .vhdr files found for subject={subject_dir.name} session={session} under {subject_dir}"
        )

    print(f"\nAvailable tasks for {subject_dir.name} ses{session}:")
    for i, t in enumerate(available, start=1):
        print(f"  {i:>2}. {t}")

    ans = input("Select task(s) by name or index (comma-separated), or 'all': ").strip().lower()
    if ans in ("all", "*"):
        return available

    chosen: List[str] = []
    for tok in [x.strip() for x in ans.split(",") if x.strip()]:
        if tok.isdigit():
            idx = int(tok) - 1
            if idx < 0 or idx >= len(available):
                raise ValueError(f"Task index out of range: {tok}")
            chosen.append(available[idx])
        else:
            if tok not in [a.lower() for a in available]:
                if tok not in available:
                    raise ValueError(f"Unknown task '{tok}'. Available: {available}")
                chosen.append(tok)
            else:
                chosen.append(next(a for a in available if a.lower() == tok))

    seen = set()
    out: List[str] = []
    for t in chosen:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def main() -> None:
    args = parse_args()
    skipped = [] #for the files that dont have triggers
    # ------------------------------------------------------------------
    # NEW: data root now defaults to BASE (derived from HUMMEL_ROOT),
    #      unless --data-root is provided.
    # ------------------------------------------------------------------
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else BASE
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}\n"
            f"Derived BASE was: {BASE}\n"
            f"If needed, pass --data-root explicitly or set HUMMEL_DATA."
        )

    subjects = _parse_subjects(args.subjects)
    if not subjects:
        one = input("Enter subject(s) (e.g., 41 or 41,42 or 41Y03): ").strip()
        subjects = _parse_subjects(one)

    sessions = _parse_sessions(args.sessions)
    tasks_arg = _parse_csv_tokens(args.tasks)

    for subj in subjects:
        subject_dir = resolve_subject_dir(data_root, subj)

        for ses in sessions:
            tasks: Sequence[str]
            if tasks_arg:
                tasks = tasks_arg
            else:
                tasks = _choose_tasks_interactive(subject_dir, session=ses)

            for task in tasks:
                vhdr_path = build_vhdr_path(subject_dir, session=ses, task=task)
                if not vhdr_path.exists():
                    raise FileNotFoundError(f"Missing input file: {vhdr_path}")

                print(f"\nProcessing: {vhdr_path}")

                try:
                    raw_concat, windows = crop_resample_concatenate(
                        vhdr_path=vhdr_path,
                        start_code=args.start_code,
                        end_code=args.end_code,
                        post_end_sec=float(args.post_end_sec),
                        target_sfreq=float(args.target_sfreq) if args.target_sfreq else None,
                        require_s8_before_next_s4=(not args.allow_s8_after_next_s4),
                        verbose="ERROR",
                        cache_root=Path.cwd() / ".bv_cache",   

                    )
                except NoBlockMarkersError as e:
                    # Skip files that do not have the required block markers
                    skipped.append({
                        "file": str(vhdr_path),
                        "reason": f"missing/unpairable markers: {e.n_start}×{e.start_code}, {e.n_end}×{e.end_code}",
                    })
                    print(f"  [skip] {vhdr_path.name}: {skipped[-1]['reason']}")
                    continue

                print(f"  Detected blocks: {len(windows)}")
                for i, w in enumerate(windows, start=1):
                    dur = w.tmax - w.tmin
                    print(f"    Block {i:02d}: t=[{w.tmin:.3f}, {w.tmax:.3f}]  (dur={dur:.3f}s)")

                if args.dry_run:
                    continue

                out_dir = Path.cwd() / args.out_folder / subject_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)

                out_fif = out_dir / f"{subject_dir.name}_ses{ses}_{task}_{args.suffix}_raw.fif"
                if out_fif.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists: {out_fif}. Use --overwrite to replace.")

                raw_concat.save(out_fif, overwrite=True)
                print(f"  Saved: {out_fif}")
    if skipped:
        print("\nSummary: files not cropped due to missing S4/S8 markers")
        for item in skipped:
            print(f"  - {item['file']}  ({item['reason']})")
    else:
        print("\nSummary: all processed files contained valid S4/S8 markers")

    print("\nDone.")


if __name__ == "__main__":
    main()
