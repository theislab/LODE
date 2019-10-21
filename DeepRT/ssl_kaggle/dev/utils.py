"""General utility functions"""

import json
import matplotlib.pyplot as plt
from PIL import Image
import glob as glob
import pandas as pd
import numpy as np
import os
import shutil

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__





class Logging():

    def __init__(self, logging_directory, params):
        self.log_dir = logging_directory
        self.model_directory = None
        self.tensorboard_directory = None
        self.params = params

    def __create_dir(self, dir):
        os.makedirs(dir)

    def __create_tensorboard_dir(self, model_dir):

        # set abs path to new dir
        new_dir = os.path.join(model_dir, "tensorboard_dir")

        # create new dir
        self.__create_dir(new_dir)

        # set object instance to new path
        self.tensorboard_directory = new_dir

    def __remove_empty_directories(self):

        # get current directories
        current_directories = glob.glob(self.log_dir + "/*")

        # check for each dir, if weight.hdf5 file is contained
        for current_directory in current_directories:
            if not os.path.isfile(os.path.join(current_directory, "weights.hdf5")):
                # remove directory
                shutil.rmtree(current_directory)

    def create_model_directory(self):
        '''
        :param logging_directory: string, gen directory for logging
        :return: None
        '''

        # remove emtpy directories
        self.__remove_empty_directories()

        # get allready created directories
        existing_ = os.listdir(self.log_dir)

        # if first model iteration, set to zero
        if existing_ == []:
            new = 0
            # save abs path of created dir
            created_dir = os.path.join(self.log_dir, str(new))

            # make new directory
            self.__create_dir(created_dir)

            # create subdir for tensorboard logs
            self.__create_tensorboard_dir(created_dir)

        else:
            # determine the new model directory
            last_ = max(list(map(int, existing_)))
            new = int(last_) + 1

            # save abs path of created dir
            created_dir = os.path.join(self.log_dir, str(new))

            # make new directory
            self.__create_dir(created_dir)

            # create subdir for tensorboard logs
            self.__create_tensorboard_dir(created_dir)

        # set class instancy to hold abs path
        self.model_directory = created_dir

    def save_dict_to_json(self, json_path):
        """Saves dict of floats in json file
        Args:
            d: (dict) of float-castable values (np.float, int, float, etc.)
            json_path: (string) path to json file
        """
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: str(v) for k, v in self.params.dict.items()}
            json.dump(d, f, indent=4)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class Evaluation():
    '''
    labels: list, integers
    predictions: list, integers
    history: pandas data frame
    '''

    def __init__(self, labels,predictions, history, model_dir, filenames, params):
        self.params = params
        self.labels = labels
        self.prediction = predictions
        self.history = history
        self.model_dir = model_dir
        self.filenames = filenames


        self.accuracy = None
        self.precision = None
        self.recall = None
        self.confusion_matrix = None

    def __accuracy(self):
        return (accuracy_score(self.labels, self.prediction))

    def __precision(self):
        return(precision_score(self.labels,self.prediction,average='micro'))

    def __recall(self):
        return(recall_score(self.labels, self.prediction,average='micro'))

    def __confusion_matrix(self):
        return(confusion_matrix(self.labels, self.prediction))

    def __filenames(self):
        # generate example predictions
        pred_im = pd.DataFrame(self.filenames)
        pred_im_pd = pred_im[0].str.split("/", expand=True)
        pred_im_pd = pred_im_pd.rename(columns={0: "labels", 1: "id"})
        return(pred_im_pd)

    def __plot_confusion_matrix(self,normalize=True,title=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        y_true = self.labels
        y_pred = self.prediction
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.matshow(cm, cmap=plt.cm.Blues)

        thresh = cm.max() / 1.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),size="large",
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.title("confusion matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.model_dir,"confusion_matrix.png"))
        return

    def __plot_history(self):
        plt.rcParams.update({'font.size': 16})
        f, axs = plt.subplots(2, 2, figsize=(10, 10))

        # load loss curves
        statistics_pd = self.history

        if 'lr' in statistics_pd:
            plt.suptitle("Train statistics")
            for i in range(1, 4):
                plt.subplot(3, 1, i)
                if i == 1:
                    plt.plot(statistics_pd["loss"], label="train loss")
                    plt.plot(statistics_pd["val_loss"], label="validation loss")
                    plt.xlabel("epochs")
                    plt.ylabel("cross entropy")
                    plt.legend()
                if i == 2:
                    plt.plot(statistics_pd["acc"], label="train accuracy")
                    plt.plot(statistics_pd["val_acc"], label="validation accuracy")
                    plt.xlabel("epochs")
                    plt.ylabel("accuracy")
                    plt.legend()
                if i == 3:
                    plt.plot(statistics_pd["lr"], label="learning rate decay")
                    plt.xlabel("epochs")
                    plt.ylabel("lr")
                    plt.legend()

        #plor without learning rate
        else:
            plt.suptitle("Train statistics")
            for i in range(1, 3):
                plt.subplot(2, 1, i)
                if i == 1:
                    plt.plot(statistics_pd["loss"], label="train loss")
                    plt.plot(statistics_pd["val_loss"], label="validation loss")
                    plt.xlabel("epochs")
                    plt.ylabel("cross entropy")
                    plt.legend()
                if i == 2:
                    plt.plot(statistics_pd["acc"], label="train accuracy")
                    plt.plot(statistics_pd["val_acc"], label="validation accuracy")
                    plt.xlabel("epochs")
                    plt.ylabel("accuracy")
                    plt.legend()


        plt.savefig(self.model_dir + "/history.png")

    def __save_example_predictions(self, params):

        #data frame with filenames and labels of test predictions
        pred_im_pd = self.__filenames()

        #only take the names of which we have predictions
        if pred_im_pd.shape[0] > len(self.prediction):
            pred_im_pd = pred_im_pd.iloc[:len(self.prediction)]

        #test prediction added
        pred_im_pd["predictions"] = self.prediction

        #set label levels
        levels = ["0", "1", "2", "3", "4"]

        for level in levels:
            pred_im_class_pd = pred_im_pd[pred_im_pd["labels"] == level]

            # shuffle indices
            pred_im_class_pd = pred_im_class_pd.sample(frac=1)

            # save ten predictions
            ten_im = pred_im_class_pd.iloc[0:5]

            for im_name in ten_im["id"]:
                pred_class = ten_im[ten_im["id"] == im_name].predictions.values[0]
                im_path = os.path.join(params.data_path, "test", level, im_name)

                # create save directory if does not exist
                if not os.path.exists(os.path.join(self.model_dir, "predictions", level)):
                    os.makedirs(os.path.join(self.model_dir, "predictions", level))

                outcome_string = "__true__" + str(level) + "__pred__" + str(pred_class) + ".jpeg"
                save_example_name = im_name.replace(".jpeg", outcome_string)

                fundus_im = np.array(Image.open(im_path))

                plt.imsave(os.path.join(self.model_dir, "predictions", level, save_example_name), fundus_im)

    def __example_prediction_canvas(self):
        plt.rcParams.update({'font.size': 5})
        example_prediction_paths = glob.glob(self.model_dir + "/predictions/**/*")

        fig = plt.figure(figsize=(10, 10))
        # set figure proportion after number of examples created
        columns = int(len(example_prediction_paths) / 5)
        rows = 5
        for i in range(1, columns * rows + 1):
            img = np.array(Image.open(example_prediction_paths[i - 1]))
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            plt.title(example_prediction_paths[i - 1].split("/")[-1].replace(".jpeg", ""))
            plt.axis('off')

        plt.savefig(os.path.join(self.model_dir, "example_canvas.png"))

    def __main_result(self):
        '''init all metrics'''
        self.accuracy = self.__accuracy()
        self.precision = self.__precision()
        self.recall = self.__recall()

        # dump all stats in txt file
        result_array = np.array(["accuracy",self.accuracy, "precision", self.precision, "recall",self.recall])
        np.savetxt(self.model_dir + "/result.txt", result_array, fmt='%s')

    def write_plot_evaluation(self):
        self.__main_result()
        self.__plot_confusion_matrix()
        self.__plot_history()

    def plot_examples(self):
        self.__save_example_predictions(self.params)
        self.__example_prediction_canvas()


