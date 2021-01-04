import os
from pprint import pprint
from keras import backend as K

import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import sys
import numpy as np

path = Path(os.getcwd())
sys.path.append(str(path.parent))

# add children paths
for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
    sys.path.append(str(child_dir))

from custom_metrics import ModelMetrics
from model_logging import ModelCheckpointCustom
from print_stats import PrintStats
from tensorboard_callback import TensorboardCallback
from losses import get_loss
from optimizers import get_optimizer

from models.model import get_model
from segmentation_config import TRAIN_DATA_PATH
from utils.utils import Params, TrainOps, Logging, data_split
from generator_2d import DataGenerator

params = Params("params.json")
params.data_path = TRAIN_DATA_PATH

logging = Logging("./logs", params)
trainops = TrainOps(params)

ids = os.listdir(os.path.join(params.data_path, "images"))
train_ids, validation_ids, test_ids = data_split(ids, params)

logging.create_model_directory()
params.model_directory = logging.model_directory

# saving model config file to model output dir
logging.save_dict_to_json(logging.model_directory + "/config.json")

# Generators
train_generator = DataGenerator(train_ids[0:1]*5, params = params, is_training = True)
validation_generator = DataGenerator(validation_ids[0:1], params = params, is_training = False)

optimizer = get_optimizer(params, trainops)
loss_fn = get_loss(params)

model_metrics = ModelMetrics(params)
tb_callback = TensorboardCallback(model_dir = params.model_directory)
model_checkpoint = ModelCheckpointCustom(monitor = "val_acc", model_dir = params.model_directory, mode = "max")
print_stats = PrintStats(params = params)

# get model
model = get_model(params)

for epoch in range(params.num_epochs):
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_generator)):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training = True)
            loss = loss_fn(y_batch_train, logits)  # focal_loss_fixed(y_batch_train, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        current_lr = optimizer._decayed_lr(tf.float32).numpy()

        print(f"\nOpt Iteration: {optimizer.__dict__['_iterations'].numpy()} "
              f"learning rate: {np.round(current_lr, 5)} loss: {np.round(loss.numpy(), 2)}")

        # Update training metric.
        model_metrics.update_metric_states(y_batch_train, logits, mode = "train")

    # Display metrics at the end of each epoch.
    train_result_dict = model_metrics.result_metrics(mode = "train")

    tb_callback.on_epoch_end(epoch = epoch, logging_dict = train_result_dict, lr = current_lr)

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in validation_generator:
        val_logits = model(x_batch_val, training = False)
        val_loss = loss_fn(y_batch_val, val_logits)

        # Update val metrics
        model_metrics.update_metric_states(y_batch_val, val_logits, mode = "val")

    print("validation loss is: ", np.round(val_loss.numpy(), 2))

    val_result_dict = model_metrics.result_metrics(mode = "val")

    tb_callback.on_epoch_end(epoch = epoch, logging_dict = val_result_dict)
    model_checkpoint.on_epoch_end(epoch, model, logging_dict = val_result_dict)
    print_stats.on_epoch_end(epoch, train_dict = train_result_dict, validation_dict = val_result_dict,
                             lr = current_lr)

    # Reset training metrics at the end of each epoch
    model_metrics.reset_metric_states(mode = "train")
    model_metrics.reset_metric_states(mode = "val")
