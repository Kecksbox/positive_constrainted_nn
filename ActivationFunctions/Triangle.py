import tensorflow as tf


class Triangle:
    def __init__(self, center: float):
        self.center = center

    def __call__(self, x, *args, **kwargs) -> tf.Tensor:
        a = tf.abs(x - self.center)
        b = tf.maximum(self.center - a, 0.0)

        return b