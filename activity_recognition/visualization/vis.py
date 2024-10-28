import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img


def visualize(model, checkpoint, manager):
    checkpoint.restore(manager.latest_checkpoint) # checkpoint restore model
    # if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))

    image_string = tf.io.read_file('IDRID_dataset/images/test/IDRiD_102.jpg') # image loading
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32) / 255.0
    image = tf.image.resize_with_pad(image, 256, 256) # preprocessing image
    last_conv_layer_index = len(model.layers) - 6  # get index of last conv layer
    last_conv_layer = model.get_layer(index=last_conv_layer_index) # get last conv layer
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output) # create model from start to last conv layer

    # Build model on top of existing Model, which can visualize the gradients
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])  #  get output shape of training model
    x = classifier_input
    for layer_index in range(last_conv_layer_index, (len(model.layers)-1)): # uses remaining layers of training model to create classifier model
        x = model.get_layer(index=layer_index)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:  # record gradients with new model
        inputs = image[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)  # get last_conv_layer_model model output
        preds = classifier_model(last_conv_layer_output)  # use classifier model to make prediction using output of last_conv_layer_model
        top_pred_index = tf.argmax(preds[0])  # get the largest activation
        top_class_channel = preds[:, top_pred_index]  # get channel of largest activation

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # average the gradients over all channels (pooling)

    last_conv_layer_output = last_conv_layer_output.numpy()[0]  # get numpy object of last conv output
    pooled_grads = pooled_grads.numpy()  # get numpy object of pooled gradients
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]  # multiply weights (pooled gradients) with output of last conv layer feature maps

    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU) and normalise the values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (256, 256))

    plt.imshow(image)
    plt.imshow(gradcam, alpha=0.5)  # set transparency of gradients to 50%
    plt.show()

