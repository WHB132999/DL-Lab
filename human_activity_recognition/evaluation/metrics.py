import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...
        self.false_nega_and_true_posi = self.add_weight(name='FNandTP', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # ...
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.math.logical_xor(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.false_nega_and_true_posi.assign_add(tf.reduce_sum(values))

    def result(self):
        # ...
        return self.false_nega_and_true_posi


# import tensorflow as tf
#
# class ConfusionMatrix(tf.keras.metrics.Metric):
#
#     def __init(self, name="confusion_matrix", **kwargs):
#         super(ConfusionMatrix, self).__init__(name=name, **kwargs)
#         # ...
#
#     def update_state(self, *args, **kwargs):
#         # ...
#
#     def result(self):
#         # ...
