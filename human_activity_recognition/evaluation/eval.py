import tensorflow as tf
import numpy as np
import pandas as pd


def evaluate(model, checkpoint, manager, ds_test, ds_info, run_paths):
    checkpoint.restore(manager.latest_checkpoint)
    # if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))

    test_iters = 0
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    for test_images, test_labels in ds_test:
        y_pred = model(test_images, training=False)
        test_iters += 1
        t_loss = loss_object(test_labels, y_pred)

        test_loss(t_loss)
        test_accuracy(test_labels, y_pred)

    if test_iters != 0:
        print("Test loss: {:.5}".format(test_loss.result()))
        print("Test Accuracy: {:.5}".format(test_accuracy.result() * 100))

    index_list = np.arange(1, 104)  # 104
    tn = 0  # label = 0, pred = 0
    tp = 0  # label = 1, pred = 1
    fn = 0  # label = 1, pred = 0
    fp = 0  # label = 0, pred = 1
    for index in index_list:
        image_index = "{:03d}".format(index)

        image_string = tf.io.read_file('IDRID_dataset/images/test/IDRiD_' + image_index + '.jpg')  # image loading
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32) / 255.0
        image = tf.image.resize_with_pad(image, 256, 256)  # preprocessing image

        directory_labels = 'IDRID_dataset/labels/'
        df_test = pd.read_csv(directory_labels + 'test.csv')
        labels_test = df_test['Retinopathy grade'].values
        label = labels_test[index - 1]
        if label < 2:
            label = 0
        elif label > 1:
            label = 1

        image_test = np.expand_dims(image, axis=0)
        y_pred = model(image_test, training=False)
        y_pred = np.argmax(tf.nn.softmax(y_pred))

        if label == 0:
            if y_pred == 0:
                tn += 1
            elif y_pred == 1:
                fp += 1
        elif label == 1:
            if y_pred == 0:
                fn += 1
            elif y_pred == 1:
                tp += 1
    print("True Positive (label = 1, pred = 1) = " + str(tp))
    print("False Positive (label = 0, pred = 1) = " + str(fp))
    print("False Negative (label = 1, pred = 0) = " + str(fn))
    print("True Negative (label = 0, pred = 0) = " + str(tn))
