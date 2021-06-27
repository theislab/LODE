from pathlib import Path
import numpy as np
import cv2
import keras

from utils import plot_segmentation


class Segmentor:
    def __init__(self, model_directory, n_models):
        """
        Parameters
        ----------
        model_directory : Model path or ensemble directory
        n_models : number of models to use for segmentation. If number is higher the # models available,
        then all available models are used.
        """
        self.model_directory = model_directory
        self.model_paths = []
        self.model_dict = {}
        self.n_models = n_models

        # current model is trained with 256, 256 images
        self.size = 256

        # if directory to model ensemble is passed, load all paths
        if not Path(model_directory).is_file():
            self.model_paths = self.model_abs_paths
        else:
            self.model_paths = [Path(model_directory)]

    @property
    def model_abs_paths(self):
        """
        Property of the Segmentor class getting the model abs paths in an ensemble directory.
        Returns
        -------
        all model paths in ensemble directory
        """

        # currently keras models in .hdf5 file format are used, could be changes in future
        if not [i for i in Path(self.model_directory).glob("*/*hdf5")]:
            raise Exception("no models are in directory")
        else:
            # select n models to use from disk
            model_paths = [i for i in Path(self.model_directory).glob("*/*hdf5")]
            n_available_models = len(model_paths)

            # if n_models is larger than available models, then all available models are used
            if n_available_models < self.n_models:
                print(f"{self.n_models} are not available, selecting {n_available_models}")

            n_models = min([n_available_models, self.n_models])
            return model_paths[0:n_models]

    def preprocess_image(self, img):
        """
        Parameters
        Functions implement expected normalizations and reshaping for model inference
        ----------
        img : array; oct image

        Returns
        -------

        """
        # resize to fit expected model input
        img_resized = cv2.resize(img, (self.size, self.size))

        # normalize
        img_normalized = img_resized / 255.

        # reshape to model input format
        img_reshaped = img_normalized.reshape(1, self.size, self.size, 3)
        return img_reshaped

    def load_keras_model(self, path):
        """
        Parameters
        ----------
        path : str; path to keras model

        Returns
        -------

        """
        return keras.models.load_model(path)

    def load_model(self):
        """
        function loads either a model or an ensemble of models into model_dict
        Returns
        -------

        """
        for k, model_path in enumerate(self.model_paths):
            self.model_dict[k] = self.load_keras_model(model_path)

    def predict(self, img):
        """
        given input image, this function returns a oct segmentation

        Parameters
        ----------
        img : array; pre processed oct image

        Returns
        -------
        semantic segmentation of oct image
        """
        ensemble_predictions = np.zeros([1, 256, 256, 16])

        for model in self.model_dict.keys():
            ensemble_predictions += self.model_dict[model].predict(img)

        # average soft max probs
        ensemble_predictions /= len(self.model_dict)

        # get final seg map
        ensemble_prediction = np.argmax(ensemble_predictions, -1)[0, :]
        return ensemble_prediction

    def segment(self, img):
        """
        this function takes a read & unprocessed oct image and return the segmentation
        Parameters
        ----------
        img : array; oct image

        Returns
        -------

        """
        preprocessed_image = self.preprocess_image(img)

        # if model is loaded, then skip loading
        if len(self.model_dict) == 0:
            # load model
            self.load_model()

        segmentation = self.predict(preprocessed_image)
        return segmentation


if __name__ == "__main__":
    model_directory = "/home/olle/PycharmProjects/LODE/feature_segmentation/ensembles_stratified"

    segmentor = Segmentor(model_directory, n_models = 1)

    # load own image
    img_path = "/home/olle/PycharmProjects/LODE/feature_segmentation/label_conversion/iteration_idv_ben/images/114421_R_20160121_23.png"
    oct_ = cv2.imread(img_path)

    seg_map = segmentor.segment(oct_)

    import matplotlib.pyplot as plt
    plot = plot_segmentation(seg_map, show_legend=True, show_legend_text=True)
    plt.show()