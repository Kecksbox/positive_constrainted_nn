from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

experiment_id = "cKpm1Dg9Sk6v2YZZFyQ0ZA"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)

df = experiment.get_scalars(pivot=False)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
dfw_val_accuracy = df[df["tag"] == 'val_accuracy']
optimizer_validation = dfw_val_accuracy.run.apply(lambda run: run.split("adam")[1])
sns.lineplot(data=dfw_val_accuracy, x="step", y="value",
             hue=optimizer_validation).set_title("accuracy")

plt.subplot(1, 2, 2)
dfw_val_loss = df[df["tag"] == 'val_loss']
optimizer_validation = dfw_val_loss.run.apply(lambda run: run.split("adam")[1])
sns.lineplot(data=dfw_val_loss, x="step", y="value",
             hue=optimizer_validation).set_title("loss")

plt.savefig('foo.png')

