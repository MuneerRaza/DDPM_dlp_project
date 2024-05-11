import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from data_preprocessing import DataPreprocessor

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
            ds.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .shuffle(self.batch_size * 2)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds
    

if __name__ == "__main__":
    dataset_name = "cifar10"
    splits = ["train"]
    batch_size = 64
    img_size = 32
    clip_min = -1.0
    clip_max = 1.0

    dataloader = DataLoader(
        dataset_name=dataset_name,
        splits=splits,
        img_size=img_size,
        batch_size=batch_size,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    train_ds = dataloader.load_data()
    print(train_ds)
