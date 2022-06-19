import tensorflow as tf
import tensorflow_probability as tfp


class NormalSign(tf.keras.initializers.Initializer):

    def __init__(self, elements_in_sum: int, act_center: float, x_mean: float = 1.0, mean: float = 1.0, x_variance=0,
                 x_squared_mean=0, y_variance=0, x=0, epsilon=1e-7):
        assert mean > 0
        self.mean = mean

        self.elements_in_sum = elements_in_sum
        self.epsilon = epsilon

        self.x_mean = x_mean
        self.act_center = act_center

        self.x_variance = x_variance
        self.x_squared_mean = x_squared_mean
        self.y_variance = y_variance

        self.x = x

    def __call__(self, shape, dtype=None, **kwargs):

        weight_mean = self.act_center / (self.x_mean * self.elements_in_sum)

        #a = (self.y_variance / self.elements_in_sum) - tf.math.square(weight_mean) * self.x_variance
        #b = self.x_variance + tf.math.square(self.x_mean)
        #variance = tf.math.divide_no_nan(a, b)

        #std = tf.math.sqrt(variance)

        weights = tfp.distributions.TruncatedNormal(
            weight_mean,
            0.05,
            low=0,
            high=2*weight_mean,
        ).sample(shape)

        # weights = tf.maximum(weights, 0.0)

        return weights
