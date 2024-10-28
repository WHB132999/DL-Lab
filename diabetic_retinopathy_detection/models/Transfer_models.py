import gin
from tensorflow.keras import Sequential, layers, Model, Input
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, MobileNetV2


class MyMobileNetV2(Model):
    """
    Define our MobileNetV2 model with the new classifier
    """
    def __init__(self, n_classes=2):
        super(MyMobileNetV2, self).__init__()
        self.total_model = Sequential([self.base_model()])
        self.total_model.add(layers.GlobalAveragePooling2D())
        self.total_model.add(layers.Dropout(rate=0.25))
        self.total_model.add(layers.Dense(n_classes))

    def base_model(self):
        # Import the MobileNetV2 model with the weights imagenet and without top
        base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
        # Set the model unable to be trainable
        base_model.trainable = False

        return base_model

    def call(self, inputs, training=False):
        outputs = self.total_model(inputs)

        return outputs


@gin.configurable
def inception_resnet_v2():
    """
    Define our InceptionResNetV2 model with the new classifier
    """
    # Import the InceptionResNetV2 model with the weights imagenet and without top
    base_model = InceptionResNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    # Set the model unable to be trainable
    base_model.trainable = False
    inputs = Input(shape=(256, 256, 3))
    # Using the new classifier
    outputs = classifier(base_model, inputs)

    return Model(inputs=inputs, outputs=outputs, name='InceptionResNetV2')


@gin.configurable
def inception_v3():
    """
    Define our InceptionV3 model with the new classifier
    """
    # Import the InceptionV3 model with the weights imagenet and without top
    base_model = InceptionV3(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    # Set the model unable to be trainable
    base_model.trainable = False
    inputs = Input(shape=(256, 256, 3))
    # Using the new classifier
    outputs = classifier(base_model, inputs)

    return Model(inputs=inputs, outputs=outputs, name='InceptionV3')


@gin.configurable
def classifier(base_model, inputs, dropout_rate, n_classes):
    """
    Define the final classifier for each different transfer model
    """
    output = base_model(inputs, training=False)
    output = layers.GlobalAveragePooling2D()(output)
    output = layers.Dropout(rate=dropout_rate)(output)
    outputs = layers.Dense(n_classes)(output)

    return outputs