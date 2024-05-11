from tensorflow import keras
import json

from network import build_model

# Load the configuration file
with open("/kaggle/working/DDPM_dlp_project/configuration/config.json") as f:
    config = json.load(f)

# with open("configuration/config.json") as f:
#     config = json.load(f)

# Get the dataset configuration
dataset_name = config["dataset"]["dataset_name"]
splits = config["dataset"]["splits"]
batch_size = config["dataset"]["batch_size"]

# Get the model configuration
img_size = config["model"]["img_size"]
img_channels = config["model"]["img_channels"]
first_conv_channels = config["model"]["first_conv_channels"]
channel_multiplier = config["model"]["channel_multiplier"]
has_attention = config["model"]["has_attention"]
num_res_blocks = config["model"]["num_res_blocks"]

widths = [first_conv_channels * mult for mult in channel_multiplier]

# Get the training configuration
total_timesteps = config["training"]["total_timesteps"]
num_epochs = config["training"]["num_epochs"]
learning_rate = config["training"]["learning_rate"]
checkpoint_dir = config["training"]["checkpoint_dir"]

resume_state = config["infer"]["resume_state"]

# Get the normalization configuration
clip_min = config["normalization"]["clip_min"]
clip_max = config["normalization"]["clip_max"]
norm_groups = config["normalization"]["norm_groups"]


model = build_model(
    img_size=img_size,
    img_channels=img_channels,
    first_conv_channels=first_conv_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    total_timesteps=total_timesteps,
)

# Load the saved weights
model.ema_network.load_weights(resume_state)

# Now you can use the model for inference
model.plot_images(num_rows=4, num_cols=8)
