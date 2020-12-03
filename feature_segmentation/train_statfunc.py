import os
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import sys

from feature_segmentation.evaluate.callbacks.metrics import ModelMetrics
from feature_segmentation.evaluate.callbacks.model_logging import ModelCheckpointCustom
from feature_segmentation.evaluate.callbacks.print_stats import PrintStats
from feature_segmentation.evaluate.callbacks.tensorboard import TensorboardCallback
from feature_segmentation.models.losses import get_loss
from feature_segmentation.models.optimizers import get_optimizer

path = Path(os.getcwd())
sys.path.append(str(path.parent))

# add children paths
for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
    sys.path.append(str(child_dir))

from models.model import get_model
from segmentation_config import TRAIN_DATA_PATH
from utils.utils import Params, TrainOps, Logging, data_split
from generator_2d import DataGenerator

params = Params("params.json")
logging = Logging("./logs", params)
trainops = TrainOps(params)

params.data_path = TRAIN_DATA_PATH

ids = os.listdir(os.path.join(params.data_path, "images"))
train_ids, validation_ids, test_ids = data_split(ids, params)

# create logging directory
logging.create_model_directory()
params.model_directory = logging.model_directory

# saving model config file to model output dir
logging.save_dict_to_json(logging.model_directory + "/config.json")

# Generators
<<<<<<< HEAD
train_generator = DataGenerator(train_ids, params=params, is_training=True)
validation_generator = DataGenerator(validation_ids, params=params, is_training=False)

# Instantiate an optimizer to train the model.logits
optimizer = keras.optimizers.SGD(learning_rate=trainops.callbacks_()[0], momentum=0.9)

# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = False)
=======
train_generator = DataGenerator(train_ids[0:1], params = params, is_training = False)
test_generator = DataGenerator(train_ids[0:1], params = params, is_training = False)

optimizer = get_optimizer(params)

# Instantiate a loss function.
loss_fn = get_loss(params)
>>>>>>> e692b8626b6ba86e89d0a4e55a9d74d64bbccd5d

model_metrics = ModelMetrics(params)
tb_callback = TensorboardCallback(model_dir = params.model_directory)
model_checkpoint = ModelCheckpointCustom(monitor="val_acc", model_dir = params.model_directory, mode="max")
print_stats = PrintStats(params=params)

# get model
model = get_model(params)

<<<<<<< HEAD
epochs = params.num_epochs
=======
>>>>>>> e692b8626b6ba86e89d0a4e55a9d74d64bbccd5d

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training = True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Update training metric.
    model_metrics.update_metric_states(y_batch_train, logits, mode = "train")


@tf.function
def test_step(x, y):
    val_logits = model(x, training = False)

    # Update val metrics
    model_metrics.update_metric_states(y, val_logits, mode = "val")


for epoch in range(params.num_epochs):
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_generator)):
        train_step(x_batch_train, y_batch_train)

    # Display metrics at the end of each epoch.
    train_result_dict = model_metrics.result_metrics(mode="train")
    tb_callback.on_epoch_end(epoch = epoch, logging_dict = train_result_dict)

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in validation_generator:
        test_step(x_batch_val, y_batch_val)

    val_result_dict = model_metrics.result_metrics(mode = "val")

    tb_callback.on_epoch_end(epoch = epoch, logging_dict = val_result_dict)
    model_checkpoint.on_epoch_end(epoch, model, logging_dict = val_result_dict)
    print_stats.on_epoch_end(epoch, train_dict=train_result_dict, validation_dict=val_result_dict)

    # Reset training metrics at the end of each epoch
    model_metrics.reset_metric_states(mode="train")
    model_metrics.reset_metric_states(mode="val")
