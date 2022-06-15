import argparse
import os
from datetime import datetime

# work_dir = "."
work_dir = os.environ["WORK"]

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

parser = argparse.ArgumentParser("photo_exp")
parser.add_argument('--log_name', type=str, default="clipping_exp_adam_abs_after")

args = parser.parse_args()

### Config ###

tf.keras.utils.set_random_seed(
    0
)

# log config
experiment_name = args.log_name
logdir = work_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + experiment_name

relative_sample_size = 0.1
near_zero_border = 0.01
sample_mask = None

# mean and std for data.
mean_input = 0.0
stddev_input = 0.05

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
activation_schema = tf.identity
kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

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

# Expand dim for mnist
# train_images, test_images = tf.expand_dims(train_images, axis=-1), tf.expand_dims(test_images, axis=-1)

# norm to center
train_images, test_images = norm_dataset(center=mean_input, std=stddev_input, train_images=train_images,
                                         test_images=test_images)


class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sequence = [
            tf.keras.layers.Conv2D(32, (3, 3), activation=activation_schema,
                                   bias_initializer=bias_initializer,
                                   kernel_initializer=kernel_initializer),
            tf.keras.layers.AveragePooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=activation_schema,
                                   bias_initializer=bias_initializer,
                                   kernel_initializer=kernel_initializer),
            tf.keras.layers.AveragePooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=activation_schema,
                                   bias_initializer=bias_initializer,
                                   kernel_initializer=kernel_initializer),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=activation_schema,
                                  bias_initializer=bias_initializer,
                                  kernel_initializer=kernel_initializer),
            tf.keras.layers.Dense(64, activation=activation_schema,
                                  bias_initializer=bias_initializer,
                                  kernel_initializer=kernel_initializer),
            tf.keras.layers.Dense(10, activation=activation_schema,
                                  bias_initializer=bias_initializer,
                                  kernel_initializer=kernel_initializer),
        ]

    def __call__(self, x, *args, **kwargs):
        for layer in self.sequence:
            x = layer(x)
        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
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

        for var in trainable_vars:
            var.assign(tf.abs(var))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


model = CustomModel()

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=False)

history = model.fit(train_images, train_labels, batch_size=64, epochs=1000, callbacks=[TensorboardCallback()],
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
