import tensorflow as tf

from ActivationFunctions.Exhibitory import Exhibitory
from ActivationFunctions.Inhibitory import Inhibitory


class ExhibitoryInhibitorySplit:
    def __init__(self, center: float):
        self.ex_mask = None

        self.inhibitory = Inhibitory(center)
        self.exhibitory = Exhibitory(center)

    def __call__(self, x, *args, **kwargs) -> tf.Tensor:
        if self.ex_mask is None:
            self.ex_mask = tf.greater(tf.random.uniform(x.shape), 0.5)

        a = self.exhibitory(x)
        b = self.inhibitory(x)
        c = tf.cast(self.ex_mask, dtype=tf.float32) * a + (1 - tf.cast(self.ex_mask, dtype=tf.float32)) * b

        return c
