import numpy as np

from feature_segmentation.evaluate.callbacks.callback_base import Callback


class PrintStats(Callback):
    def __init__(self, metrics=None, params=None):
        self.metrics = metrics
        self.params = params

        if metrics is not None:
            assert isinstance(metrics, list), "metrics to print must be passed as a list"

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch, train_dict, validation_dict):
        """
        Parameters
        ----------
        epoch :
        logging_dict :

        Returns
        -------
        """
        train_dict.update(validation_dict)
        key_values = list(train_dict.keys())

        if self.metrics is None:
            metric_str = ""
            for key in key_values:
                key_value = str(np.round(train_dict[key].numpy(), 2))
                metric_str = metric_str + f"{key} {key_value}: "

        else:
            metric_str = ""
            for key in self.metrics:
                key_value = str(np.round(train_dict[key].numpy(), 2))
                metric_str = metric_str + f"{key}: {key_value}: "
        print(f"Epoch {epoch} / {self.params.num_epochs} :: " + metric_str)