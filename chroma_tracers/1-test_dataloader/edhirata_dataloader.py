# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchview
import torchvision
from iohub import open_ome_zarr
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from skimage import metrics  # for metrics.

# HCSDataModule makes it easy to load data during training.
from viscy.light.data import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import VSTrainer, VSUNet
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard

BATCH_SIZE = 4


# %%
input_data_path = (
    "/home/eduardoh/vs_data/0-raw_data/1-H2B_dataset/target_fluorescence/deskewed.zarr"
)
log_dir = Path("/mnt/efs/shared_data/virtual_staining/1-test_dataloader/logs")


# Load the data
input_data = open_ome_zarr(input_data_path)
seed_everything(42, workers=True)
# Create log directory if needed, and launch tensorboard
log_dir.mkdir(parents=True, exist_ok=True)


# %%
data_module = HCSDataModule(
    input_data_path,
    source_channel="Phase3D",
    target_channel=["GFP EX488 EM525-45", "mCherry EX561 EM600-37"],
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="2.1D",
    yx_patch_size=(512, 512),  # larger patch size makes it easy to see augmentations.
    augment=False,  # Turn off augmentation for now.
)
# NOTE: if the `normalization does not exist`

data_module.setup("fit")
# train_dataloader = data_module.train_dataloader()
print(
    f"FOVs in training set: {len(data_module.train_dataset)}, FOVs in validation set:{len(data_module.val_dataset)}"
)
train_dataloader = data_module.train_dataloader()


# %%
def log_batch_tensorboard(batch, batchno, writer, card_name):
    """
    Logs a batch of images to TensorBoard.

    Args:
        batch (dict): A dictionary containing the batch of images to be logged.
        writer (SummaryWriter): A TensorBoard SummaryWriter object.
        card_name (str): The name of the card to be displayed in TensorBoard.

    Returns:
        None
    """
    batch_phase = batch["source"][:, :, 0, :, :]  # batch_size x z_size x Y x X tensor.
    batch_membrane = batch["target"][:, 1, 0, :, :].unsqueeze(
        1
    )  # batch_size x 1 x Y x X tensor.
    batch_nuclei = batch["target"][:, 0, 0, :, :].unsqueeze(
        1
    )  # batch_size x 1 x Y x X tensor.

    p1, p99 = np.percentile(batch_membrane, (0.1, 99.9))
    batch_membrane = np.clip((batch_membrane - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_nuclei, (0.1, 99.9))
    batch_nuclei = np.clip((batch_nuclei - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_phase, (0.1, 99.9))
    batch_phase = np.clip((batch_phase - p1) / (p99 - p1), 0, 1)

    [N, C, H, W] = batch_phase.shape
    interleaved_images = torch.zeros((3 * N, C, H, W), dtype=batch_phase.dtype)
    interleaved_images[0::3, :] = batch_phase
    interleaved_images[1::3, :] = batch_nuclei
    interleaved_images[2::3, :] = batch_membrane

    grid = torchvision.utils.make_grid(interleaved_images, nrow=3)

    # add the grid to tensorboard
    writer.add_image(card_name, grid, batchno)


# %%
writer = SummaryWriter(log_dir=f"{log_dir}/view_batch")
# Draw a batch and write to tensorboard.
batch = next(iter(train_dataloader))
log_batch_tensorboard(batch, 0, writer, "augmentation/none")
writer.close()

### View the augmentations
# Here we turn on data augmentation and rerun setup
data_module.augment = True
data_module.setup("fit")

# get the new data loader with augmentation turned on
augmented_train_dataloader = data_module.train_dataloader()

# Draw batches and write to tensorboard
writer = SummaryWriter(log_dir=f"{log_dir}/view_batch")
augmented_batch = next(iter(augmented_train_dataloader))
log_batch_tensorboard(augmented_batch, 0, writer, "augmentation/some")
writer.close()
# %%
