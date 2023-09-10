# %%
from torchview import draw_graph

from viscy.light.engine import VSUNet

# %% 2D UNet
model = VSUNet(
    architecture="2D",
    model_config={
        "in_channels": 2,
        "out_channels": 1,
        "num_filters": [24, 48, 96, 192, 384],
    },
)

model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="2D UNet",
    roll=True,
    depth=2,
    # graph_dir="LR",
    # save_graph=True,
)

graph2d = model_graph.visual_graph
graph2d

# %% 2.5D UNet
model = VSUNet(
    architecture="2.5D",
    model_config={
        "in_channels": 1,
        "out_channels": 3,
        "in_stack_depth": 9,
        "num_filters": [24, 48, 96, 192, 384],
    },
)

model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="2.5D UNet",
    roll=True,
    depth=2,
)

graph25d = model_graph.visual_graph
graph25d

# %%
# %% 2.5D LUNet
model = VSUNet(
    architecture="25D_LUnet",
    model_config={
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 5,
        "num_filters": [24, 48, 96, 192, 384],
    },
)

model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="2.D_LUnet",
    roll=True,
    depth=3,
)

graph25d_LUnet = model_graph.visual_graph
graph25d_LUnet

# %%
