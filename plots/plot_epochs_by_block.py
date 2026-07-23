import mne
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

derivatives_dir = "/work/uphummel/studies/tTIS-EEG/derivatives/EEG"
subject = "41Y01"
session = "1"
task = "task"
mode = "surface"
method = "MNE"
label_kind = "labels"
epoch_kind = "task_fixed"
n_sources = 20

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "epoched_ROI_time_courses"
output_dir.mkdir(parents=True, exist_ok=True)

base_tag = f"{subject}_ses{session}_task-{task}_src-{mode}_method-{method}"
epochs_path = (
    f"{derivatives_dir}/sub-{subject}/ses-{session}/{task}/epochs/{mode}/{method}/"
    f"{base_tag}_{label_kind}_{epoch_kind}-epo.fif"
)

epochs = mne.read_epochs(
    epochs_path,
    preload=True,
)

print(f"Opening: {epochs_path}")
print(epochs.metadata[["condition", "block", "epoch_in_block", "duration_s"]])

for block in sorted(epochs.metadata["block"].unique()):
    block_epochs = epochs[epochs.metadata["block"] == block]

    data = block_epochs.get_data()
    times = block_epochs.times
    average_data = data[:, :n_sources, :].mean(axis=0)

    plt.figure(figsize=(12, 6))
    for i, roi in enumerate(block_epochs.ch_names[:n_sources]):
        plt.plot(times, average_data[i], label=roi)

    plt.title(f"Block {block} average across {len(block_epochs)} epochs")
    plt.xlabel("Time (s)")
    plt.ylabel("ROI time course")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path = output_dir / f"{base_tag}_{epoch_kind}_block_{block}_average.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
