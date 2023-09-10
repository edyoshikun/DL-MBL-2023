"""
pip install iohub

you can use the command-line (activate your conda environment where you installed iohub)

If you use the command line:

# This shows the options throught the CLI:
iohub convert --help 

#This would reconstruct in your command line
iohub convert -i /path/to/input/directory -o "/path/to/output/directory.zarr"

"""

from iohub.ngff import open_ome_zarr
from iohub.cli.cli import convert

# Set up the arguments as needed
input_dir = "/path/to/input/directory"
output_zarr_dir = "/path/to/output/directory" + ".zarr"


# Logic to read their tiff files with `tiffffile`
input_dataset = tifffile.imread(input_dir)  # assuming it can be held in RAM
Z, Y, X = input_dataset.shape

# Chunk size selector
# TODO: convenient function to make chunks <500MB
chunk_zyx_shape = list(Z, Y, X)
# chunk_zyx_shape[-3] > 1 ensures while loop will not stall if single
# XY image is larger than MAX_CHUNK_SIZE
while (
    chunk_zyx_shape[-3] > 1
    and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
):
    chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
chunk_zyx_shape = tuple(chunk_zyx_shape)
chunk_size = 2 * (1,) + chunk_zyx_shape


# Call the convert function with the specified arguments
with open_ome_zarr(intput_dir, mode="hcs", mode="w") as output_dataset:
    pos = output_dataset.create_position(
        str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
    )
    position = pos.create_zeros(
        name="0",
        shape=output_shape,
        chunks=chunk_size,
        dtype=DTYPE,
        transform=[transform],
    )
    for t_idx in range(T):
        for c_idx in range(C):
            position[0][T,C] = 

