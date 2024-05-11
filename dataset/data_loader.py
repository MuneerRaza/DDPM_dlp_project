import tensorflow as tf
import tensorflow_datasets as tfds
import json

from .data_preprocessing import DataPreprocessor

class DataLoader:
    def __init__(self, dataset_name, splits, img_size, batch_size, clip_min, clip_max):
        self.dataset_name = dataset_name
        self.splits = splits
        self.img_size = img_size
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def load_data(self):
        # Load the dataset
        (ds,) = tfds.load(self.dataset_name, split=self.splits, with_info=False, shuffle_files=True)

        # Initialize the DataPreprocessor class
        train_preprocessing = DataPreprocessor(self.img_size, self.clip_min, self.clip_max)

        train_ds = (
            ds.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .shuffle(self.batch_size * 2)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds
