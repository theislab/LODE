import numpy as np
import os

from feature_segmentation.evaluate.callbacks.callback_base import Callback


class ModelCheckpointCustom(Callback):
    def __init__(self, model_dir=None, monitor="val_loss", mode="max"):
        self.loss_values = []
        self.metric_values = []
        self.model_dir = model_dir
        self.monitor = monitor
        self.mode = mode

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch, model, logging_dict):
        """
        Parameters
        ----------
        epoch :
        logging_dict :

        Returns
        -------
        """
        self.loss_values.append(logging_dict[self.monitor].numpy())

        if self.mode == "max":
            def compare(current_value, previous_losses):
                return current_value > max(previous_losses), max(previous_losses)
        else:
            def compare(current_value, previous_losses):
                return current_value < min(previous_losses), min(previous_losses)

        if len(self.loss_values) >= 2:
            improved, previous_best = compare(logging_dict[self.monitor].numpy(), self.loss_values[0:-1])
            if improved:
                previous_best = str(np.round(previous_best, 2))
                current = str(np.round(logging_dict[self.monitor].numpy(), 2))
                print(f"#### Model validation {self.monitor} improved from {previous_best} to "
                      f"{current}, saving model in {self.model_dir}####")

                model.save(os.path.join(self.model_dir, "model.h5"))
            else:
                print(f"---> Model validation {self.monitor} did not improve, continue without saving the model")

