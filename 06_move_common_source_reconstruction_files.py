#!/usr/bin/env python3
# Author Maelys Clerget / OpenAI Codex
"""
Move shared source-reconstruction files into their session-level locations.

Personalized montage:
  sub-*/ses-*/personalized_montage/

Coregistration transform:
  sub-*/ses-*/source_reconstruction/

This is intentionally separate from the forward/inverse scripts so those jobs
do not spend time reorganizing files while they run on the cluster.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

DERIVATIVES_DIR = "/work/uphummel/studies/tTIS-EEG/derivatives/EEG"


def chmod_group(path: Path) -> None:
    try:
        os.chmod(path, 0o770)
    except PermissionError:
        print(f"Could not chmod {path}; continuing.")


def chmod_group_dir(path: Path) -> None:
    try:
        os.chmod(path, 0o2770)
    except PermissionError:
        print(f"Could not chmod {path}; continuing.")


def move_file(src: Path, dst: Path, dry_run: bool, overwrite: bool) -> None:
    if not src.exists():
        return

    if src.resolve() == dst.resolve():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    chmod_group_dir(dst.parent)

    if dst.exists() and not overwrite:
        print(f"Keeping existing destination, source left in place: {dst}")
        return

    action = "Would move" if dry_run else "Moving"
    print(f"{action}: {src} -> {dst}")

    if not dry_run:
        shutil.move(str(src), str(dst))
        chmod_group(dst)


def organize_subject_session(subject: str, session: str, dry_run: bool, overwrite: bool) -> None:
    ses_dir = Path(DERIVATIVES_DIR) / f"sub-{subject}" / f"ses-{session}"
    source_recon_dir = ses_dir / "source_reconstruction"

    old_montage_dir = source_recon_dir / "personalized_montage"
    new_montage_dir = ses_dir / "personalized_montage"
    montage_name = f"{subject}_ses{session}_personalized_montage"

    for suffix in (".fif", ".png"):
        move_file(
            old_montage_dir / f"{montage_name}{suffix}",
            new_montage_dir / f"{montage_name}{suffix}",
            dry_run,
            overwrite,
        )

    trans_name = f"{subject}_ses{session}_trans.fif"
    target_trans = source_recon_dir / trans_name
    for src in sorted(source_recon_dir.glob(f"**/{trans_name}")):
        move_file(src, target_trans, dry_run, overwrite)


def iter_subject_sessions(subject: str | None, session: str | None):
    root = Path(DERIVATIVES_DIR)
    if subject and session:
        yield subject, session
        return

    for sub_dir in sorted(root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_dir.name.removeprefix("sub-")
        if subject and sub != subject:
            continue
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses = ses_dir.name.removeprefix("ses-")
            if session and ses != session:
                continue
            yield sub, ses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move montage and trans files into the shared session-level source-reconstruction layout."
    )
    parser.add_argument("--subject", default=None, help="Optional subject ID without sub- prefix.")
    parser.add_argument("--session", default=None, help="Optional session ID without ses- prefix.")
    parser.add_argument("--derivatives-dir", default=DERIVATIVES_DIR, help="EEG derivatives root.")
    parser.add_argument("--dry-run", action="store_true", help="Print moves without changing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DERIVATIVES_DIR = args.derivatives_dir
    for subject, session in iter_subject_sessions(args.subject, args.session):
        organize_subject_session(subject, session, args.dry_run, args.overwrite)
