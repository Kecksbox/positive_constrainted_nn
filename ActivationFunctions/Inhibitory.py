import tensorflow as tf


class Inhibitory:
    def __init__(self, center: float):
        self.center = center

    def __call__(self, x, *args, **kwargs) -> tf.Tensor:
        return tf.abs(tf.minimum(x - self.center, 0.0))
