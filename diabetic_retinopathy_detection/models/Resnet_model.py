import gin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, regularizers


def regularized_conv2d(*args, **kwargs):
    """
    Build a defined regularized convolutional layer for better training and used for the ResNet
    Using batch normalization after each convolutional layer, hence use_bias=False
    Kernel_initializer by default is glorot_uniform, but he_normal is better for training deeper NN
    Kernel_regularizer we use l2 regularization
    """
    return layers.Conv2D(
        *args, **kwargs, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5)
    )


class BasicBlock(layers.Layer):
    """
    Define the basic block
    """
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional unit
        self.conv1 = regularized_conv2d(filter_num, kernel_size=[3, 3], strides=stride)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        # Second convolutional unit
        self.conv2 = regularized_conv2d(filter_num, kernel_size=[3, 3], strides=1)
        self.bn2 = layers.BatchNormalization()
        # Define the by pass for the basic block
        if stride != 1:
            self.downsample = Sequential([
                regularized_conv2d(filter_num, kernel_size=[1, 1], strides=stride),
                layers.BatchNormalization()
            ])
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        identity = self.downsample(inputs)
        # Sum up the output of the basic block and the input
        output = tf.nn.relu(layers.add([identity, output]))
        return output


@gin.configurable
class ResNet(keras.Model):
    """
    Define the basic architecture of the ResNet
    """
    def __init__(self, layer_dims, num_classes):
        super(ResNet, self).__init__()
        # Input layer
        self.stem = Sequential([
                                layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                                regularized_conv2d(64, kernel_size=[3, 3], strides=1),
                                layers.BatchNormalization(),
                                layers.ReLU()
        ])
        # Build four blocks, each block has several BasicBlock and with different strides
        self.layer1 = self.build_resblock(64, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def build_resblock(self, filter_num, blocks, stride=1):
        """  Build several BasicBlocks for one resblock  """
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    def call(self, inputs, training=False):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x


@gin.configurable
def resnet18(num_classes):
    """  Build ResNet18  """
    return ResNet(layer_dims=[2, 2, 2, 2], num_classes=num_classes)


@gin.configurable
def resnet34(num_classes):
    """  Build ResNet34  """
    return ResNet(layer_dims=[3, 4, 6, 3], num_classes=num_classes)











# from tensorflow.keras import Sequential, Dense, Flatten, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D
#
#
# def cnn_lstm_model(window_size):
#     model = Sequential()
#     model.add(TimeDistributed(
#         Conv1D(filters=32, kernel_size=3, activation='relu'),
#         input_shape=(None, window_size, 6)))
#     model.add(TimeDistributed(
#         Conv1D(filters=64, kernel_size=3, activation='relu')))
#     model.add(TimeDistributed(Dropout(0.5)))
#     model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
#     model.add(TimeDistributed(Flatten()))
#     model.add(LSTM(100))
#     model.add(Dropout(0.5))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(6, activation='softmax'))
#
#     return model
#
# from tensorflow.keras import Sequential