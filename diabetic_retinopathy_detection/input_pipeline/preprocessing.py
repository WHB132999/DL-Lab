import gin
import tensorflow as tf

# THIS PART WAS CREATED BY SAMUEL BRUCKER

@gin.configurable
def preprocess(image, img_height, img_width, sobel_preprocessing):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize_with_pad(image, len(image[1]), len(image[1]))
    image = tf.image.central_crop(image, central_fraction=0.7)
    image = tf.image.resize_with_pad(image, img_height, img_width)

    # SOBEL EDGE FILTER
    if sobel_preprocessing is True:
        image = tf.expand_dims(image, 0)
        image = tf.image.sobel_edges(image)
        image = image[0, ..., 0] / 4 + 0.5

    return image


@gin.configurable
def augment(image, label):
    """Data augmentation"""

    # convert to grayscale with a probability of 10%
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    # change brightness with a maximum delta of 0.1
    image = tf.image.random_brightness(image, max_delta=0.05)

    # flip image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # random crop image with probability of 50%
    if tf.random.uniform((), minval=0, maxval=1) < 0.5:
        image_dimension = len(image[0])
        image = tf.image.random_crop(image, size=[round(image_dimension * 0.95), round(image_dimension * 0.95), 3])
        image = tf.image.resize_with_pad(image, image_dimension, image_dimension)

    return image, label