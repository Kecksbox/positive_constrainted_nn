import tensorflow as tf


@tf.function
def log_grad(x):
    return tf.gradients(tf.math.log(x), [x])


class SGDSign(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate, name="SGDSignOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        epsilon = 1e-7
        # curvature_factor = tf.maximum(tf.math.divide_no_nan(1.0, log_grad(var)), 1e-7)
        # curvature_factor = tf.reshape(curvature_factor, tf.shape(grad))
        # grad = grad * curvature_factor

        new_var = tf.abs(var - lr_t * grad)

        var.assign(new_var)
