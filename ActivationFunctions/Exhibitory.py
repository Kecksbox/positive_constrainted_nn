import tensorflow as tf


class Exhibitory:
    def __init__(self, center: float):
        self.center = center

    def __call__(self, x, *args, **kwargs) -> tf.Tensor:
        return tf.abs(tf.maximum(x - self.center, 0.0))
