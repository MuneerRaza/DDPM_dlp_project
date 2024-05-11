import tensorflow as tf


class DataPreprocessor:
    def __init__(self, img_size, clip_min, clip_max):
        self.img_size = img_size
        self.clip_min = clip_min
        self.clip_max = clip_max



    def augment(self, img):
        """Flips an image left/right randomly."""
        return tf.image.random_flip_left_right(img)


    def resize_and_rescale(self, img):
        """Resize the image to the desired size first and then
        rescale the pixel values in the range [-1.0, 1.0].

        Args:
            img: Image tensor
            size: Desired image size for resizing
        Returns:
            Resized and rescaled image tensor
        """

        height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        crop_size = tf.minimum(height, width)

        img = tf.image.crop_to_bounding_box(
            img,
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )

        # Resize
        img = tf.cast(img, dtype=tf.float32)
        img = tf.image.resize(img, size=(self.img_size, self.img_size), antialias=True)

        # Rescale the pixel values
        img = img / 127.5 - 1.0
        img = tf.clip_by_value(img, self.clip_min, self.clip_max)
        return img


    def train_preprocessing(self, x):
        img = x["image"]
        img = self.resize_and_rescale(img)
        img = self.augment(img)
        return img