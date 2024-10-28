import gin
import tensorflow as tf
import numpy as np

# THIS PART WAS PARTLY CREATED BY HUIBIN WANG AND PARTLY BY SAMUEL BRUCKER

@gin.configurable
def evaluate(model, checkpoint, manager, ds_test):
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("No checkpoints found, evaluation not possible.")

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

    # Confusion Matrix
    tn = 0  # label = 0, predicted = 0
    tp = 0  # label = 1, predicted = 1
    fn = 0  # label = 1, predicted = 0
    fp = 0  # label = 0, predicted = 1
    for test_images, test_labels in ds_test:
        y_pred = model(test_images, training=False)
        y_pred = np.argmax(tf.nn.softmax(y_pred))
        if test_labels == 0:
            if y_pred == 0:
                tn += 1
            elif y_pred == 1:
                fp += 1
        elif test_labels == 1:
            if y_pred == 0:
                fn += 1
            elif y_pred == 1:
                tp += 1

    print("---- Confusion Matrix -----")
    print("True Positives (label = 1, pred = 1) = " + str(tp))
    print("False Positives (label = 0, pred = 1) = " + str(fp))
    print("False Negatives (label = 1, pred = 0) = " + str(fn))
    print("True Negatives (label = 0, pred = 0) = " + str(tn))
    print("---------------------------")