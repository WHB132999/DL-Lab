import gin
import tensorflow as tf

@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize_with_pad(image, img_height, img_width)
    # image = tf.image.resize(image, size=(img_height, img_width))

    return image, label

@gin.configurable
def augment(image, label): #doesnt work with img_height, img_width? !?!?!??
    """Data augmentation"""

    # convert to grayscale with a probability of 10% !!
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    # change brightness with a maximum delta of 0.1
    image = tf.image.random_brightness(image, max_delta=0.05)

    # change contrast
    # image = tf.image.random_contrast(image, lower=0.1, upper=0.2)

    # flip image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # central crop image with probability of 20%
    if tf.random.uniform((), minval=0, maxval=1) < 0.5:
        image = tf.image.central_crop(image, central_fraction=0.7)
        image = tf.image.resize_with_pad(image, 256, 256)

    return image, label