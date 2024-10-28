import tensorflow as tf



class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...
        self.correct_results = self.add_weight(name='correct_results', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # ...
        y_pred = tf.argmax(y_pred, axis=1)
        y_pred = tf.cast(y_pred, dtype=tf.int32)
        y_true = tf.cast(y_true, dtype=tf.int32)
        correct = tf.equal(y_true, y_pred)
        correct = tf.cast(correct, dtype=tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, correct.shape)
            correct = tf.multiply(correct, sample_weight)

        self.correct_results.assign_add(tf.reduce_sum(correct))

    def result(self):
        # ...
        return self.correct_results  # Hint: only sum up all correct classifications, haven't divide with the total
                                     # number of train/test samples, namely, not percentage result, so it will be very large.
                                     # Because I don't know how many samples for train and test.


# import tensorflow as tf
#
#
# class ConfusionMatrix(tf.keras.metrics.Metric):
#
#     def __init(self, name="confusion_matrix", **kwargs):
#         super(ConfusionMatrix, self).__init__(name=name, **kwargs)
#         # ...
#         self.false_nega_and_true_posi = self.add_weight(name='FNandTP', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # ...
#         y_true = tf.cast(y_true, tf.bool)
#         y_pred = tf.cast(y_pred, tf.bool)
#
#         values = tf.math.logical_xor(tf.equal(y_true, True), tf.equal(y_pred, False))
#         values = tf.cast(values, self.dtype)
#         if sample_weight is not None:
#             sample_weight = tf.cast(sample_weight, self.dtype)
#             sample_weight = tf.broadcast_to(sample_weight, values.shape)
#             values = tf.multiply(values, sample_weight)
#         self.false_nega_and_true_posi.assign_add(tf.reduce_sum(values))
#
#     def result(self):
#         # ...
#         return self.false_nega_and_true_posi