import gin
import tensorflow as tf
# import keras as k
# from keras import layers, Sequential

from models.layers import vgg_block


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate
    Returns:
        (keras.Model): keras model object
    """
    assert n_blocks > 0, 'Number of blocks has to be at least 1.'
    inputs = tf.keras.Input(shape=(250, 6), batch_size=16)  # Input shape is (batch_size, 256, 256, 3)
    out = vgg_block(inputs, base_filters, kernel_size=3, strides=1)  # (batch_size, 256, 256, 3)->(batch_size, 128, 128, 32)
    for i in range(1, n_blocks):
        out = vgg_block(out, base_filters * (2 ** i), kernel_size=3, strides=1)  # (batch_size, 16, 16, 256)
    out = tf.keras.layers.GlobalAveragePooling1D()(out)  # (batch_size, 256)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(6)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


# The inherit way to build our model.


# class VggBlock(layers.Layer):
#     def __init__(self, filters, kernel_size, strides=1):
#         super(VggBlock, self).__init__()
#         self.conv1 = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')
#         self.bn1 = layers.BatchNormalization()
#         self.relu1 = layers.Activation('relu')
#         self.conv2 = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')
#         self.bn2 = layers.BatchNormalization()
#         self.relu2 = layers.Activation('relu')
#         self.max_pool = layers.MaxPool2D((2, 2))
#         self.dropout = layers.Dropout(rate=0.25, training=True)
#
#     def call(self, inputs, training=False):
#         out = self.conv1(inputs)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu2(out)
#         out = self.max_pool(out)
#         output = self.dropout(out)
#
#         return output
#
#
# def build_block(n_blocks, filters, kernel_size):
#     whole_block = Sequential()
#     whole_block.add(VggBlock(filters=filters, kernel_size=kernel_size))
#     for i in range(1, n_blocks):
#         whole_block.add(VggBlock(filters=filters * (2 ** i), kernel_size=kernel_size))
#
#     return whole_block
#
#
# class MyModel(k.Model):
#     def __init__(self, n_blocks, filters, kernel_size, dense_units, n_classes, strides=1):
#         super(MyModel, self).__init__()
#         self.stem = build_block(n_blocks=n_blocks, filters= filters, kernel_size=kernel_size)
#         self.global_pool = layers.GlobalAveragePooling2D()
#         self.dense1 = layers.Dense(dense_units, activation=tf.nn.relu)
#         self.dense2 = layers.Dense(n_classes)
#
#     def call(self, inputs, training=False):
#         out = self.stem(inputs)
#         out = self.global_pool(out)
#         out = self.dense1(out)
#         output = self.dense2(out)
#
#         return output
#
#
# my_new_model = MyModel(n_blocks=4, filters=32, kernel_size=(3, 3), dense_units=256, n_classes=2)
# my_new_model.summary()

# import gin
# import tensorflow as tf
#
# from models.layers import vgg_block
#
# @gin.configurable
# def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
#     """Defines a VGG-like architecture.
#
#     Parameters:
#         input_shape (tuple: 3): input shape of the neural network
#         n_classes (int): number of classes, corresponding to the number of output neurons
#         base_filters (int): number of base filters, which are doubled for every VGG block
#         n_blocks (int): number of VGG blocks
#         dense_units (int): number of dense units
#         dropout_rate (float): dropout rate
#
#     Returns:
#         (keras.Model): keras model object
#     """
#
#     assert n_blocks > 0, 'Number of blocks has to be at least 1.'
#
#     inputs = tf.keras.Input(input_shape)
#     out = vgg_block(inputs, base_filters)
#     for i in range(2, n_blocks):
#         out = vgg_block(out, base_filters * 2 ** (i))
#     out = tf.keras.layers.GlobalAveragePooling2D()(out)
#     out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
#     out = tf.keras.layers.Dropout(dropout_rate)(out)
#     outputs = tf.keras.layers.Dense(n_classes)(out)
#
#     return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')