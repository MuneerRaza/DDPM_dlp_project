import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

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
        datapreprocessor = DataPreprocessor(self.img_size, self.clip_min, self.clip_max)

        train_preprocessing = lambda x: datapreprocessor.train_preprocessing(x)
        train_ds = (
            tqdm(ds.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE))
            .batch(self.batch_size, drop_remainder=True)
            .shuffle(self.batch_size * 2)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds
