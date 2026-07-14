#!/usr/bin/env python3

import mne
import numpy as np

cov4_path = (
    "/work/uphummel/studies/tTIS-EEG/derivatives/EEG/"
    "sub-41Y01/ses-1/source_reconstruction/inverse_solution/"
    "volume/eLORETA/task/covs4"
    "41Y01_ses1_task-task_src-volume_method-eLORETA_cov-s4-cov.fif"
)

cov15_path = (
    "/work/uphummel/studies/tTIS-EEG/derivatives/EEG/"
    "sub-41Y01/ses-1/source_reconstruction/inverse_solution/"
    "volume/eLORETA/task/covs15/"
    "41Y01_ses1_task-task_src-volume_method-eLORETA_cov-s15-cov.fif"
)

cov4 = mne.read_cov(cov4_path)
cov15 = mne.read_cov(cov15_path)

C4 = cov4.data
C15 = cov15.data

print("cov4:", cov4_path)
print("cov15:", cov15_path)
print("same shape:", C4.shape, C15.shape)

if C4.shape != C15.shape:
    raise SystemExit("Cannot compare values because shapes differ.")

diff = C4 - C15

print("exactly equal:", np.array_equal(C4, C15))
print("almost equal:", np.allclose(C4, C15))
print("max absolute difference:", np.max(np.abs(diff)))
print("mean absolute difference:", np.mean(np.abs(diff)))
print("relative difference:", np.linalg.norm(diff) / np.linalg.norm(C4))