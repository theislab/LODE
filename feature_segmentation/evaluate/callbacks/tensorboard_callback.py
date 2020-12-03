import datetime
import os
import tensorflow as tf

from feature_segmentation.evaluate.callbacks.callback_base import Callback


class TensorboardCallback(Callback):
    def __init__(self, model_dir=None):
        log_dir = os.path.join(model_dir, "tensorboard_dir")
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch, logging_dict, lr=None):
        """
        Parameters
        ----------
        epoch :
        logging_dict :

        Returns
        -------

        """

        for key, item in logging_dict.items():
            with self.summary_writer.as_default():
                tf.summary.scalar(key, logging_dict[key], step = epoch)
                tf.summary.scalar(key, logging_dict[key], step = epoch)

        if lr is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("lr", lr, step = epoch)
