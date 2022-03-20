"""Metrics for TF training.

Classes
---------
InterquartileCapture(tf.keras.metrics.Metric)
SignTest(tf.keras.metrics.Metric)
CustomMAE(tf.keras.metrics.Metric)


"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class InterquartileCapture(tf.keras.metrics.Metric):
    """Compute the fraction of true values between the 25 and 75 percentiles.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight("count", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
        mu = pred[:, 0]
        sigma = pred[:, 1]
        norm_dist = tfp.distributions.Normal(mu,sigma)
        lower = norm_dist.quantile(.25)
        upper = norm_dist.quantile(.75)

        batch_count = tf.reduce_sum(
            tf.cast(
                tf.math.logical_and(
                    tf.math.greater(y_true[:, 0], lower),
                    tf.math.less(y_true[:, 0], upper)
                ),
                tf.float32
            )

        )
        batch_total = len(y_true[:, 0])

        self.count.assign_add(tf.cast(batch_count, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.count / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class SignTest(tf.keras.metrics.Metric):
    """Compute the fraction of true values above the median.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight("count", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
        mu = pred[:, 0]
        sigma = pred[:, 1]
        norm_dist = tfp.distributions.Normal(mu,sigma)
        median = norm_dist.quantile(.50)

        batch_count = tf.reduce_sum(
            tf.cast(tf.math.greater(y_true[:, 0], median), tf.float32)
        )
        batch_total = len(y_true[:, 0])

        self.count.assign_add(tf.cast(batch_count, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.count / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    
class CustomMAE(tf.keras.metrics.Metric):
    """Compute the prediction mean absolute error.

    The "predicted value" is the median of the conditional distribution.

    Notes
    -----
    * The computation is done by maintaining running sums of total predictions
        and correct predictions made across all batches in an epoch. The
        running sums are reset at the end of each epoch.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error = self.add_weight("error", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
        mu = pred[:, 0]
        sigma = pred[:, 1]
        norm_dist = tfp.distributions.Normal(mu,sigma)
        predictions = norm_dist.quantile(.50)

        error = tf.math.abs(y_true[:, 0] - predictions)
        batch_error = tf.reduce_sum(error)
        batch_total = tf.math.count_nonzero(error)

        self.error.assign_add(tf.cast(batch_error, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.error / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}