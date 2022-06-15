import tensorflow as tf

from ActivationFunctions.Exhibitory import Exhibitory
from ActivationFunctions.Inhibitory import Inhibitory


class ExhibitoryInhibitory:
    def __init__(self, center: float, alpha: float):
        self.alpha = alpha
        self.center = center

        self.inhibitory = Inhibitory(center)
        self.exhibitory = Exhibitory(center)

    def __call__(self, x, *args, **kwargs) -> tf.Tensor:
        a = self.exhibitory(x)
        b = self.inhibitory(x)
        c = self.alpha * b + a

        return c