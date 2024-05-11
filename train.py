
import json
import os
import re
from tensorflow import keras
from tqdm import tqdm
from tqdm.keras import TqdmCallback

from network import build_model
from dataset.data_loader import DataLoader

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
checkpoint_period = config["training"]["checkpoint_period"]
resume_state = config["training"]["resume_state"]

# Get the normalization configuration
clip_min = config["normalization"]["clip_min"]
clip_max = config["normalization"]["clip_max"]
norm_groups = config["normalization"]["norm_groups"]

# Load the dataset
dataloader = DataLoader(
    dataset_name=dataset_name,
    splits=splits,
    img_size=img_size,
    batch_size=batch_size,
    clip_min=clip_min,
    clip_max=clip_max,
)

# Build the model
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

# Compile the model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

train_ds = dataloader.load_data()
tqdm_callback = TqdmCallback()

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Path where to save the model
path_checkpoint = os.path.join(checkpoint_dir, "Model_{epoch:04d}.h5")

cp_callback = None
if checkpoint_period:
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=path_checkpoint,
        save_weights_only=True,
        save_freq=checkpoint_period,
    )

if resume_state:
    model.load_weights(resume_state)
    last_epoch = re.search(r'Model_(\d{4})\.h5', resume_state).group(1)

    # Calculate the remaining epochs
    num_epochs = num_epochs - last_epoch

    print(f"Resuming from {last_epoch} epoch.")

    # Train the model
model.fit(
    train_ds,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[cp_callback, tqdm_callback] if checkpoint_period else [tqdm_callback],
    verbose=0,
)
    

# Save the model
model.save_weights(path_checkpoint.format(epoch=num_epochs))

