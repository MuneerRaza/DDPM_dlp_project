import tensorflow as tf
import tensorflow_datasets as tfds

from config import batch_size, img_size, dataset_name, splits, clip_min, clip_max
from data_preprocessing import DataPreprocessor

# Load the dataset
(ds,) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)

# Initialize the DataPreprocessor class
train_preprocessing = DataPreprocessor(img_size, clip_min, clip_max)

train_ds = (
    ds.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=True)
    .shuffle(batch_size * 2)
    .prefetch(tf.data.AUTOTUNE)
)