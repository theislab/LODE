import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from models.callbacks.custom_metrics import ModelMetrics
from models.callbacks.model_logging import ModelCheckpointCustom
from models.callbacks.print_stats import PrintStats
from models.callbacks.tensorboard_callback import TensorboardCallback
from models.losses import get_loss
from models.optimizers import get_optimizer

from models.model import get_model
from config import TRAIN_DATA_PATH, DATA_SPLIT_PATH
from utils.utils import Params, TrainOps, Logging
from generator import DataGenerator


def main(flags):
    params = Params("params.json")
    params.data_path = TRAIN_DATA_PATH

    params.cv_iteration = flags.cfs_cv_iteration

    logging = Logging(flags.save_model_dir, params)

    train_ids = pd.read_csv(DATA_SPLIT_PATH + "/train_ids.csv")["0"].tolist()
    validation_ids = pd.read_csv(DATA_SPLIT_PATH + "/validation_ids.csv")["0"].tolist()
    test_ids = pd.read_csv(DATA_SPLIT_PATH + "/test_ids.csv")["0"].tolist()

    test_id = [test_ids[params.cv_iteration]]

    # log test id
    params.test_id = test_id[0]

    print("Test records is: ", test_id[0])

    test_ids = [id_ for id_ in test_ids if id_ not in test_id]
    extra_ids = test_ids
    random.shuffle(extra_ids)

    train_ids = train_ids + extra_ids[0: int(len(extra_ids) * 0.75)]
    validation_ids = validation_ids + extra_ids[int(len(extra_ids) * 0.75):]

    print(f"Number of training samples: {len(train_ids)}, "
          f"number of validation samples: {len(validation_ids)}, "
          f"number of test sample: {len(test_id)}")

    logging.create_model_directory(model_dir=f"{flags.save_model_dir}/{test_id[0].replace('.png', '')}")
    params.model_directory = logging.model_directory

    # saving model config file to model output dir
    logging.save_dict_to_json(logging.model_directory + "/config.json")

    # Generators
    train_generator = DataGenerator(train_ids, params=params, is_training=True)
    validation_generator = DataGenerator(validation_ids, params=params, is_training=False)

    trainops = TrainOps(params, num_records=len(train_generator))

    optimizer = get_optimizer(params, trainops)
    loss_fn = get_loss(params)

    model_metrics = ModelMetrics(params)
    tb_callback = TensorboardCallback(model_dir=params.model_directory)
    model_checkpoint = ModelCheckpointCustom(monitor="val_acc", model_dir=params.model_directory, mode="max")
    print_stats = PrintStats(params=params)

    # get model
    model = get_model(params)

    for epoch in range(params.num_epochs):
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_generator)):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            current_lr = optimizer._decayed_lr(tf.float32).numpy()
            print(f"\nOpt Iteration: {optimizer.__dict__['_iterations'].numpy()} "
                  f"learning rate: {current_lr} loss: {np.round(loss.numpy(), 2):.2f}")

            # Update training metric.
            model_metrics.update_metric_states(y_batch_train, logits, mode="train")

        # Display metrics at the end of each epoch.
        train_result_dict = model_metrics.result_metrics(mode="train")

        tb_callback.on_epoch_end(epoch=epoch, logging_dict=train_result_dict, lr=current_lr)

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in validation_generator:
            val_logits = model(x_batch_val, training=False)
            val_loss = loss_fn(y_batch_val, val_logits)

            # Update val metrics
            model_metrics.update_metric_states(y_batch_val, val_logits, mode="val")

        print(f"validation loss is: f'{val_loss.numpy():.2f}'")

        val_result_dict = model_metrics.result_metrics(mode="val")

        tb_callback.on_epoch_end(epoch=epoch, logging_dict=val_result_dict)
        model_checkpoint.on_epoch_end(epoch, model, logging_dict=val_result_dict)
        print_stats.on_epoch_end(epoch, train_dict=train_result_dict, validation_dict=val_result_dict,
                                 lr=current_lr)

        # Reset training metrics at the end of each epoch
        model_metrics.reset_metric_states(mode="train")
        model_metrics.reset_metric_states(mode="val")


if __name__ == "__main__":
<<<<<<< HEAD
    NUM_ENSEMBLES = 1
=======
    NUM_ENSEMBLES = 5
>>>>>>> ef05297d95903ce7425ed5038983580209fec0fd
    NUM_OCTS = 35

    class FLAGS:
        pass

    flags = FLAGS()
    for j in range(NUM_ENSEMBLES):
        for i in range(NUM_OCTS):
            flags.cfs_cv_iteration = i
<<<<<<< HEAD
            flags.save_model_dir = f"./cv_logs1/logs_{j}"
=======

            # set model dir where much disk space is available. With CV training many models will be saved.
            flags.save_model_dir = f"./logs_{j}"
>>>>>>> ef05297d95903ce7425ed5038983580209fec0fd
            main(flags)
