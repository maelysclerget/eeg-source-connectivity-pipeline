import mne
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

n_sources = 20

epochs = mne.read_epochs(
    "/work/uphummel/studies/tTIS-EEG/derivatives/EEG/sub-41Y01/ses-1/task/epochs/surface/MNE/41Y01_ses1_task-task_src-surface_method-MNE_labels_task_fixed-epo.fif",
    preload=True,
)

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
    plt.savefig(f"block_{block}_average.png", dpi=300)
    plt.close()
