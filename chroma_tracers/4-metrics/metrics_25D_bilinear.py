# %%
import viscy

viscy.__file__

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
from viscy.light.engine import MixedLoss
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard

# HCSDataModule makes it easy to load data during training.
from viscy.light.data import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import VSTrainer, VSUNet

seed_everything(42, workers=True)

import pandas as pd
from skimage import metrics  # for metrics.

import os
import napari

os.environ["DISPLAY"] = ":1"
viewer = napari.Viewer()


GPU_ID = 0
YX_PATCH_SIZE = ((1564 // 16) * 16, (1176 // 16) * 16)
BATCH_SIZE = 4
NUM_WORKERS = 16

# Paths to data and log directory
data_path = Path(Path("/home/eduardoh/cropped_dataset_v3.zarr"))
assert data_path.exists()

# Dictionary that specifies key parameters of the model.
phase2fluor_25D_LUnet_config = {
    "num_filters": [48, 48, 96, 192, 384],
    "in_channels": 1,
    "out_channels": 2,
    "in_stack_depth": 5,
    "residual": True,
    "dropout": 0.1,  # dropout randomly turns off weights to avoid overfitting of the model to data.
    "task": "reg",  # reg = regression task.
}
phase2fluor_25D_LUnet_model = VSUNet(
    "2.5D",
    model_config=phase2fluor_25D_LUnet_config.copy(),
    loss_function=torch.nn.functional.l1_loss,
    schedule="WarmupCosine",
    log_num_samples=3,  # Number of samples from each batch to log to tensorboard.
    example_input_yx_shape=YX_PATCH_SIZE,
    lr=2e-4,
)
# Reinitialize the data module.
phase2fluor_data = HCSDataModule(
    data_path,
    source_channel="Phase3D",
    target_channel=["GFP EX488 EM525-45", "mCherry EX561 EM600-37"],
    z_window_size=5,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    architecture="25D_LUnet",
    yx_patch_size=YX_PATCH_SIZE,
    normalize_source=True,
    augment=True,
    train_z_scale_range=[-0.2, 0.5],
    train_patches_per_stack=2,
    train_noise_std=2.0,
)

print(phase2fluor_data)
phase2fluor_data.setup("fit")
train_dataloader = phase2fluor_data.train_dataloader()
val_dataloader = phase2fluor_data.val_dataloader()

test_metrics = pd.DataFrame(
    columns=["pearson_nuc", "SSIM_nuc", "pearson_mem", "SSIM_mem"]
)


def min_max_scale(input):
    return (input - np.min(input)) / (np.max(input) - np.min(input))


ckpt_fpath = "/mnt/efs/shared_data/virtual_staining/model_checkpoints/Unet21D_AR_adhoc-epoch50.ckpt"
checkpoint = torch.load(ckpt_fpath)
_ = phase2fluor_25D_LUnet_model.load_state_dict(checkpoint["state_dict"])

# %%
# for i, sample_train in enumerate(train_dataloader):
#     break
# print(sample_train["target"].shape)

# # %%
# for i, sample in enumerate(val_dataloader):
#     break

# with torch.inference_mode():
#     predicted_image = phase2fluor_25D_LUnet_model(sample["source"])

# %%
pred_stack = []
target_stack = []
# Made the batchsize to 1 so we predict
for i, sample in enumerate(val_dataloader):
    if i == 20:
        break
    print(f"processing {i}")
    phase_image = sample["source"]
    with torch.inference_mode():  # turn off gradient computation.
        predicted_image = phase2fluor_25D_LUnet_model(phase_image)
    target_image = sample["target"].cpu().numpy()
    predicted_image = predicted_image.cpu().numpy()
    # print(f"predictionshape: {predicted_image.shape}")
    phase_image = phase_image.cpu().numpy()
    B, C, Z, Y, X = target_image.shape
    for b_idx in range(B):
        target_mem = min_max_scale(target_image[b_idx, 1, slice(Z // 2, Z // 2 + 1)])
        target_nuc = min_max_scale(target_image[b_idx, 0, slice(Z // 2, Z // 2 + 1)])
        # print(f" minmax target mem: {target_mem.shape}, nuc {target_nuc.shape}")
        # slicing channel dimension, squeezing z-dimension.
        predicted_mem = min_max_scale(predicted_image[b_idx, 1, 0, :, :])
        predicted_nuc = min_max_scale(predicted_image[b_idx, 0, 0, :, :])

        # Compute SSIM and pearson correlation.
        ssim_nuc = metrics.structural_similarity(
            target_nuc[0], predicted_nuc, data_range=target_nuc.max() - target_nuc.min()
        )
        ssim_mem = metrics.structural_similarity(
            target_mem[0], predicted_mem, data_range=target_mem.max() - target_mem.min()
        )
        pearson_nuc = np.corrcoef(target_nuc.flatten(), predicted_nuc.flatten())[0, 1]
        pearson_mem = np.corrcoef(target_mem.flatten(), predicted_mem.flatten())[0, 1]
        # print(target_image.shape, target_image[b_idx].max(), predicted_mem.max(), predicted_nuc.max())

        test_metrics.loc[i * B + b_idx] = {
            "pearson_nuc": pearson_nuc,
            "SSIM_nuc": ssim_nuc,
            "pearson_mem": pearson_mem,
            "SSIM_mem": ssim_mem,
        }

        pred_stack.append(np.stack((predicted_nuc, predicted_mem)))
        target_stack.append(np.stack((target_nuc, target_mem)))

pred_stack = np.array(pred_stack)
target_stack = np.array(target_stack)
# %%
viewer.add_image(pred_stack)
viewer.add_image(target_stack)
# %%
import tifffile

tifffile.imwrite("./pred_stack", pred_stack)
tifffile.imwrite("./pred_stack", pred_stack)

out_pd_file = "/hpc/projects/comp.micro/mantis/2023_08_09_HEK_PCNA_H2B/xx-mbl_course_H2B/25dLunet_metrics.csv"
test_metrics.boxplot(
    column=["pearson_nuc", "SSIM_nuc", "pearson_mem", "SSIM_mem"],
    rot=30,
)
plt.show()
