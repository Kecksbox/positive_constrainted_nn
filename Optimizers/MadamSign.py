import tensorflow as tf


class MadamSign(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, g_bound=10.0,
                 bind_lr: bool = False,
                 name="Madam", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)

        assert learning_rate < 1

        self.step = 0
        self.g_bound = g_bound
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))  # handle lr=learning_rate

        self.bind_lr = bind_lr

        self.lr_t = learning_rate

        self.initial_lr = learning_rate

    def apply_gradients(
            self, grads_and_vars, name=None, experimental_aggregate_gradients=True
    ):

        lr_t = self._decayed_lr(tf.float32)

        # TODO: Doesn`t work that well better replace with line search
        if self.bind_lr:
            grads_and_vars = list(grads_and_vars)
            groups = dict()
            for grad, var in grads_and_vars:
                layer_name = str(var.name.split('/', 1)[0])
                if layer_name not in groups:
                    groups[layer_name] = (tf.reshape(grad, shape=(-1,)), tf.reshape(var, shape=(-1,)))
                else:
                    grad = tf.reshape(grad, shape=(-1,))
                    var = tf.reshape(var, shape=(-1,))
                    groups[layer_name] = (
                        tf.concat([groups[layer_name][0], grad], axis=0),
                        tf.concat([groups[layer_name][1], var], axis=0)
                    )

            L = len(groups)
            trust = self.initial_lr
            lr_t = float('inf')
            for grad, var in groups.values():
                abs_grad = tf.abs(grad)
                abs_var = tf.abs(var)
                dot = tf.reduce_sum(tf.reshape(abs_grad, shape=(-1,)) * tf.reshape(abs_var, shape=(-1,)))
                cos_angle = dot / (tf.norm(abs_grad) * tf.norm(abs_var))
                max_lr = tf.math.pow(1 + cos_angle, 1 / L) - 1
                if lr_t >= max_lr:
                    lr_t = trust * max_lr

        self.lr_t = lr_t

        super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "g_")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        exp_avg_sq = self.get_slot(var, "g_")

        self.step += 1
        bias_correction = 1 - 0.999 ** self.step
        exp_avg_sq.assign(0.999 * exp_avg_sq + 0.001 * grad ** 2)

        g_normed = grad / tf.math.sqrt(exp_avg_sq / bias_correction)
        g_normed_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(g_normed)), dtype=var_dtype)
        g_normed = tf.math.multiply_no_nan(g_normed, g_normed_not_nan)
        g_normed = tf.clip_by_value(g_normed, -self.g_bound, self.g_bound)

        new_var = var * tf.math.exp(-lr_t * g_normed * tf.math.sign(var))

        new_var = tf.abs(new_var)

        var.assign(new_var)
