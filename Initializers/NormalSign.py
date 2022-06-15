import tensorflow as tf
import tensorflow_probability as tfp


class NormalSign(tf.keras.initializers.Initializer):

    def __init__(self, elements_in_sum: int, act_center: float, x_mean: float = 1.0, mean: float = 1.0, x_variance=0,
                 x_squared_mean=0, y_variance=0, epsilon=1e-7):
        assert mean > 0
        self.mean = mean

        self.elements_in_sum = elements_in_sum
        self.epsilon = epsilon

        self.x_mean = x_mean
        self.act_center = act_center

        self.x_variance = x_variance
        self.x_squared_mean = x_squared_mean
        self.y_variance = y_variance

    def __call__(self, shape, dtype=None, **kwargs):

        bias = tf.maximum(self.act_center - self.x_mean, 0.0)
        weight_mean = 1 / self.elements_in_sum + tf.math.divide_no_nan(bias, (self.elements_in_sum * self.x_mean))

        a = (self.y_variance / self.elements_in_sum) - tf.math.square(weight_mean) * self.x_variance
        b = self.x_squared_mean
        variance = tf.math.divide_no_nan(a, b)

        std = tf.math.sqrt(variance)

        weights = tf.random.normal(shape, mean=weight_mean, stddev=std, dtype=dtype)

        #mask = tf.cast(tf.logical_or(tf.less(weights, 0.0), tf.greater(weights, 2 * weight_mean)), dtype=tf.float32)

        #weights = tf.random.uniform(shape, minval=0, maxval=2 * weight_mean) * mask + (1 - mask) * weights

        #weight_mean = tf.reduce_mean(weights)

        #weights -= weight_mean

        #factor = (weight_mean - tf.abs(weights)) / weight_mean

        #std_w_old_times_f = tf.math.reduce_std(weights * factor)

        #scale = std / std_w_old_times_f

        #weights = weights * factor * scale

        #weights += weight_mean

        #weights = tf.maximum(weights, 0.0)

        return weights
