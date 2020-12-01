import keras
import tensorflow as tf
from feature_segmentation.models.metrics import DiceCoefficient


class ModelMetrics:
    def __init__(self, config):
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_cross_entropy = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.val_cross_entropy = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.train_dice = DiceCoefficient(num_classes = config.num_classes)
        self.val_dice = DiceCoefficient(num_classes = config.num_classes)

    def update_metric_states(self, y, y_hat, mode):
        for key, metric in self.__dict__.items():
            if mode in key:
                metric.update_state(y, y_hat)

    def reset_metric_states(self, mode):
        for key, metric in self.__dict__.items():
            if mode in key:
                metric.reset_states()

    def result_metrics(self, mode):
        result_dict = {}
        for key, metric in self.__dict__.items():
            if mode in key:
                result_dict[key] = metric.result()
        return result_dict