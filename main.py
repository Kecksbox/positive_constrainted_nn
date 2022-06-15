import argparse
import os
from datetime import datetime
from typing import List

from keras.callbacks import ReduceLROnPlateau

work_dir = "."
# work_dir = os.environ["WORK"]

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from ActivationFunctions.Exhibitory import Exhibitory
from ActivationFunctions.ExhibitoryInhibitory import ExhibitoryInhibitory
from ActivationFunctions.Triangle import Triangle
from Initializers.NormalSign import NormalSign
from Optimizers.MadamSign import MadamSign
from Optimizers.SGDSign import SGDSign, log_grad

parser = argparse.ArgumentParser("photo_exp")
parser.add_argument('--log_name', type=str, default="soifsonsei")

args = parser.parse_args()

### Config ###

tf.keras.utils.set_random_seed(
    25235
)

use_data_augmentation = False

# log config
experiment_name = args.log_name
logdir = work_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + experiment_name

relative_sample_size = 0.1
near_zero_border = 0.01
sample_mask = None

center_act_function = 0.25  # mean_activation

# mean and std for data.
mean_input = center_act_function
stddev_input = 0.05

# mean and std for kernel.
mean_kernel = 1.0
stddev_kernel = 0.05

# mean and std for bias.
mean_bias = center_act_function
stddev_bias = 0.05

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001)  # MadamSign() tf.keras.optimizers.Adam() # SGDSign(learning_rate=0.01) tf.keras.optimizers.SGD()
activation_schema = ExhibitoryInhibitory(center=center_act_function, alpha=1.0)
kernel_initializer = NormalSign
bias_initializer = tf.constant_initializer(
    value=0.0
)


class NonNegativeAbs(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.abs(w)


# Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

###        ###

tb_summary_writer = tf.summary.create_file_writer(logdir)


class TensorboardCallback(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self):
        super(TensorboardCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        with tb_summary_writer.as_default():
            for key, value in logs.items():
                tf.summary.scalar(key, value, step=epoch)

            vars = model.trainable_variables
            for i in range(len(vars)):
                var = vars[i]
                vars[i] = tf.reshape(var, shape=(-1,))
            complete_weight_checkpoint = tf.concat(vars, axis=0)

            number_zero_weights = tf.reduce_sum(tf.cast(tf.equal(complete_weight_checkpoint, 0.0), tf.float32))
            tf.summary.scalar('zero_weights', number_zero_weights, step=epoch)

            global sample_mask
            if sample_mask is None:
                sample_mask = tf.less(tf.random.uniform(complete_weight_checkpoint.shape), relative_sample_size)

            sampled_weight_checkpoint = tf.boolean_mask(complete_weight_checkpoint, sample_mask)
            sampled_weight_checkpoint = tf.abs(sampled_weight_checkpoint)

            tf.summary.histogram("weights", sampled_weight_checkpoint, step=epoch)

            sampled_weight_checkpoint = tf.boolean_mask(
                sampled_weight_checkpoint, tf.less(sampled_weight_checkpoint, near_zero_border)
            )
            sampled_weight_checkpoint = tf.math.sign(sampled_weight_checkpoint) * tf.math.log(
                1 + tf.abs(sampled_weight_checkpoint))
            tf.summary.histogram("weights_log_scale", sampled_weight_checkpoint, step=epoch)


def norm_dataset(center, std, train_images, test_images):
    x_test_size = test_images.shape[0]

    x_all = tf.concat([train_images, test_images], axis=0)
    x_std = tf.math.reduce_std(x_all)
    x_mean = tf.math.reduce_mean(x_all)
    x_all = tf.math.divide_no_nan((x_all - x_mean), x_std)

    x_all *= std
    x_all += center

    x_all = tf.cast(tf.abs(x_all), dtype=tf.float32)

    train_images = x_all[:-x_test_size]
    test_images = x_all[-x_test_size:]

    return train_images, test_images


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# norm to center
train_images, test_images = norm_dataset(center=mean_input, std=stddev_input, train_images=train_images,
                                         test_images=test_images)

if use_data_augmentation:
    tf.print("Start DataAug. This will take a while!")
    with tf.device('cpu'):
        data_augmentation = tf.keras.layers.RandomFlip()
        train_images = tf.concat([train_images, data_augmentation(train_images)], axis=0)
        train_labels = tf.concat([train_labels, train_labels], axis=0)

        train_images = tf.random.shuffle(train_images, seed=0)
        train_labels = tf.random.shuffle(train_labels, seed=0)


class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.acts = [
            activation_schema,
            None,
            activation_schema,
            None,
            activation_schema,
            None,
            activation_schema,
            activation_schema,
            activation_schema,
        ]
        self.sequence = [
            tf.keras.layers.Conv2D(32, (3, 3), name="trainable/conv1",
                                   activation=None,
                                   kernel_constraint=NonNegativeAbs(),
                                   bias_constraint=NonNegativeAbs(),
                                   bias_initializer=tf.constant_initializer(value=0),
                                   kernel_initializer=kernel_initializer(elements_in_sum=9 * 3,
                                                                         act_center=center_act_function)),
            tf.keras.layers.AveragePooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), name="trainable/conv2",
                                   activation=None,
                                   kernel_constraint=NonNegativeAbs(),
                                   bias_constraint=NonNegativeAbs(),
                                   bias_initializer=bias_initializer,
                                   kernel_initializer=kernel_initializer(elements_in_sum=9 * 32,
                                                                         act_center=center_act_function)),
            tf.keras.layers.AveragePooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), name="trainable/conv3",
                                   activation=None,
                                   kernel_constraint=NonNegativeAbs(),
                                   bias_constraint=NonNegativeAbs(),
                                   bias_initializer=bias_initializer,
                                   kernel_initializer=kernel_initializer(elements_in_sum=9 * 64,
                                                                         act_center=center_act_function)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, name="trainable/dense1",
                                  activation=None,
                                  kernel_constraint=NonNegativeAbs(),
                                  bias_constraint=NonNegativeAbs(),
                                  bias_initializer=bias_initializer,
                                  kernel_initializer=kernel_initializer(elements_in_sum=1024,
                                                                        act_center=center_act_function)),
            tf.keras.layers.Dense(64, name="trainable/dense2",
                                  activation=None,
                                  kernel_constraint=NonNegativeAbs(),
                                  bias_constraint=NonNegativeAbs(),
                                  bias_initializer=bias_initializer,
                                  kernel_initializer=kernel_initializer(elements_in_sum=512,
                                                                        act_center=center_act_function)),
            tf.keras.layers.Dense(10, name="trainable/dense3",
                                  activation=None,
                                  kernel_constraint=NonNegativeAbs(),
                                  bias_constraint=NonNegativeAbs(),
                                  bias_initializer=bias_initializer,
                                  kernel_initializer=kernel_initializer(elements_in_sum=64,
                                                                        act_center=center_act_function)),
        ]
        self.gradient_stats = {}

    def __call__(self, x, training=False, *args, **kwargs):
        y_variance = tf.math.square(tf.math.reduce_std(x))
        for layer, act in zip(self.sequence, self.acts):
            if hasattr(layer, 'kernel_initializer') and not hasattr(layer, 'kernel'):
                x_squared_mean = tf.math.reduce_mean(tf.math.square(x))
                x_variance = tf.math.square(tf.math.reduce_std(x))
                x_mean = tf.math.reduce_mean(x)
                layer.kernel_initializer = kernel_initializer(
                    elements_in_sum=layer.kernel_initializer.elements_in_sum,
                    x_mean=x_mean,
                    act_center=center_act_function,
                    x_variance=x_variance,
                    x_squared_mean=x_squared_mean,
                    y_variance=y_variance,
                )
            x_mean = tf.math.reduce_mean(x)
            x_std = tf.math.reduce_std(x)
            x = layer(x, training=training)
            x_mean2 = tf.math.reduce_mean(x)
            x_std2 = tf.math.reduce_std(x)
            y_variance = tf.math.square(tf.math.reduce_std(x))
            if act is not None:
                x = act(x)

        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape(persistent=False) as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


model = CustomModel()

model(test_images[:4000])

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=True,
                              patience=20, min_lr=0.000001)
_ = model.fit(train_images, train_labels, batch_size=64, epochs=1000, callbacks=[TensorboardCallback(), reduce_lr],
              validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
