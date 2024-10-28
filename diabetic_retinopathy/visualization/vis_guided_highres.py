import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img


def visualize(model, checkpoint, manager):
    checkpoint.restore(manager.latest_checkpoint) # checkpoint restore model
    # if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    index_list = np.arange(1, 104) #104
    for index in index_list:
        image_index = "{:03d}".format(index)


        image_string = tf.io.read_file('IDRID_dataset/images/test/IDRiD_' + image_index + '.jpg') # image loading
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32) / 255.0
        image = tf.image.resize_with_pad(image, 256, 256) # preprocessing image

        directory_labels = 'IDRID_dataset/labels/'
        df_test = pd.read_csv(directory_labels + 'test.csv')
        labels_test = df_test['Retinopathy grade'].values
        label = labels_test[index - 1]
        if label < 2:
            label = 0
        elif label > 1:
            label = 1
        header1 = ("label = " + str(label))

        image_test = np.expand_dims(image, axis=0)
        y_pred = model(image_test, training=False)
        y_pred = np.argmax(tf.nn.softmax(y_pred))
        header2 = ("   predicted label = " + str(y_pred))

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

        last_conv_layer_output = last_conv_layer_output[0]

        guided_grads = (
                tf.cast(last_conv_layer_output > 0, "float32")
                * tf.cast(grads > 0, "float32")
                * grads
        )
     #  (0, 1) (3, 1, 0) (3, 0)
        pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(3, 1, 0))
        guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32 )

        for i, w in enumerate(pooled_guided_grads):
            guided_gradcam += w * last_conv_layer_output[:, :, i]

        guided_gradcam = cv2.resize(guided_gradcam.numpy(), (255, 255))

        guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
        guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
                guided_gradcam.max() - guided_gradcam.min()
        )
        print(image_index + 'of 103 ...')
        fig, (left, right) = plt.subplots(1, 2)
        left.imshow(image)
        right.imshow(image)
        right.imshow(guided_gradcam, alpha=0.5)
        fig.suptitle(header1 + header2)
        fig.savefig('grad-cam_outputs/image_' + image_index + '.jpg', dpi=600)

