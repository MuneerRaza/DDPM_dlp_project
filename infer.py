from tensorflow import keras

from config import img_size, img_channels, widths, has_attention, num_res_blocks, norm_groups, total_timesteps
from unet import build_model
from diffusion_utils import GaussianDiffusion
from train import DiffusionModel

# Initialize the same model architecture
network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)
ema_network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)

# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# Get the model
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

# Load the saved weights
model.ema_network.load_weights('checkpoints/diffusion_model_checkpoint')

# Now you can use the model for inference
model.plot_images(num_rows=4, num_cols=8)
