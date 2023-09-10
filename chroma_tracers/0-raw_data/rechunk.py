# %%
from iohub import open_ome_zarr
import numpy as np
from tqdm import tqdm

input_data_path = "/mnt/efs/shared_data/virtual_staining/0-raw_data/1-H2B_dataset/target_fluorescence/cropped_dataset_v3.zarr/0/0/0"

# %%
with open_ome_zarr(input_data_path) as dataset:
    dataset[0].info
