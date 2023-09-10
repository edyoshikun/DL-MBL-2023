
# %%
from iohub import open_ome_zarr
from iohub.cli.cli import convert
import tifffile
import numpy as np
from tqdm import tqdm
import os
os.environ['DISPLAY']=':1'
import napari
# %%
print("Starting reading tiffs")
data1 = tifffile.imread(
    "/mnt/efs/shared_data/virtual_staining/SIM_all/SIM_stack_tiff/tubulin201-240.tif"
)
data2 = tifffile.imread(
    "/mnt/efs/shared_data/virtual_staining/SIM_all/SIM_stack_tiff/MT201-240.tif"
)
data3 = tifffile.imread(
    "/mnt/efs/shared_data/virtual_staining/SIM_all/SIM_stack_tiff/actin201-240.tif"
)
print("Done reading tiffs")
total_cells, Y, X = data1.shape

print(data1.shape)
output_directory = "/mnt/efs/shared_data/virtual_staining/SIM_all/SIM_stack_zarr/Multi-Channel201-240.zarr"
# %%

channel_names = ["Red Tubulin", "Green MT", "Orange Actin"]
num_channels = len(channel_names)
with open_ome_zarr(
    output_directory, mode="w", layout="hcs", channel_names=channel_names
) as dataset:
    for cell_id in tqdm(range(total_cells)):
        # print("Creating empty array")
        pos = dataset.create_position("0", str(cell_id), "0")
        print("Creating empty array")
        stack = pos.create_zeros(
            name="0",
            shape=(1, num_channels, 1, Y, X),
            dtype=np.float32,
            chunks=(1, 1, 1, Y, X),
        )
        print("Copying data")
        # stack[0,:,0] = data  # (3,Y,X)
        stack[0, 0, 0] = data1[cell_id]
        stack[0, 1, 0] = data2[cell_id]
        stack[0, 2, 0] = data3[cell_id]
# %%
# with open_ome_zarr(output_directory) as output_dataset:
#     print(output_dataset["0"].shape)
#     plt.imshow(output_dataset[0][0, 0, 20])
# %%
viewer = napari.Viewer()