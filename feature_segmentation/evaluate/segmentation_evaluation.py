import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import glob


class Segmentor:
    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.model_paths = []
        self.model_dict = {}
        # if directory to model ensemble is passed, load all paths
        if not Path(model_directory).is_file():
            self.model_paths = self.get_model_paths
        else:
            self.model_paths = [Path(model_directory)]

    @property
    def get_model_paths(self):
        if not [i for i in Path(self.model_directory).glob("*/*hdf5")]:
            raise Exception("no models are in directory")
        else:
            return [i for i in Path(self.model_directory).glob("*/*hdf5")]

    def preprocess_image(self):
        pass

    def load_model(self):
        pass

    def load_ensemble(self):
        pass

    def load_segmenter(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    model_directory = "/home/olle/PycharmProjects/LODE/feature_segmentation/ensembles_stratified"

    segmentor = Segmentor(model_directory)

    # load own image

    cv2.imread()
    pass