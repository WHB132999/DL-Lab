import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from input_pipeline.preprocessing import preprocess
import glob
import os


# THIS PART WAS CREATED BY HUIBIN WANG AND SAMUEL BRUCKER

def visualize_an(model, checkpoint, manager, img_height, img_width, sobel_preprocessing):
    checkpoint.restore(manager.latest_checkpoint)  # checkpoint restore model
    # if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    for filename in glob.glob('IDRID_preprocessed/test/*.jpg'):
        index = int(filename[-7:-4])
        image_index = filename[-7:-4]

        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        image = preprocess(image_decoded, img_height, img_width, sobel_preprocessing)

        directory_labels = 'D:/Python/My Pycode/dl-lab-21w-team15/diabetic_retinopathy/IDRID_dataset/labels/'
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

        # get index of last conv layer
        last_conv_layer_index = len(model.layers) - 6
        # get last conv layer
        last_conv_layer = model.get_layer(index=last_conv_layer_index)
        # create model from start to last conv layer
        last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

        # Build model on top of existing Model, which can visualize the gradients
        #  get output shape of training model
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        # uses remaining layers of training model to create classifier model
        for layer_index in range(last_conv_layer_index, (len(model.layers) - 1)):
            x = model.get_layer(index=layer_index)(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        # record gradients with new model
        with tf.GradientTape() as tape:
            inputs = image[np.newaxis, ...]
            last_conv_layer_output = last_conv_layer_model(inputs)
            # get last_conv_layer_model model output
            tape.watch(last_conv_layer_output)
            # use classifier model to make prediction using output of last_conv_layer_model
            preds = classifier_model(last_conv_layer_output)
            # get the largest activation
            top_pred_index = tf.argmax(preds[0])
            # get channel of largest activation
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        last_conv_layer_output = last_conv_layer_output[0]

        guided_grads = (
                tf.cast(last_conv_layer_output > 0, "float32")
                * tf.cast(grads > 0, "float32")
                * grads
        )

        pooled_guided_grads = tf.reduce_mean(-1 * guided_grads, axis=(0, 1, 2))
        guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32)

        for i, w in enumerate(pooled_guided_grads):
            guided_gradcam += w * last_conv_layer_output[:, :, i]

        guided_gradcam = cv2.resize(guided_gradcam.numpy(), (255, 255))

        guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
        guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
                guided_gradcam.max() - guided_gradcam.min()
        )
        print(image_index + ' of 103 ...')
        fig, (left, right) = plt.subplots(1, 2)
        left.imshow(image)
        right.imshow(image)
        right.imshow(guided_gradcam, alpha=0.5)
        fig.suptitle(header1 + header2)
        if os.path.isdir('grad-cam_outputs_an/') is False:
            print("Grad-CAM output directory not existent yet, creating directory...")
            os.mkdir('grad-cam_outputs_an/')

        fig.savefig('grad-cam_outputs_an/image_' + image_index + '.jpg', dpi=600)