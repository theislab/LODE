"""General utility functions"""

import json
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import glob
import shutil
import tensorflow
from PIL import Image
import numpy as np
import matplotlib.gridspec as gridspec
import os
from sklearn.metrics import jaccard_score
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard


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
        self.learning_rate = None
        self.batch_size = None
        self.num_epochs = None
        self.data_path = None
        self.img_shape = None
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent = 4)

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

    def __create_main_directory(self):
        '''
        :return: create main log dir if not allready created
        '''
        if not os.path.isdir(self.log_dir):
            print("main logging dir does not exist, creating main logging dir ./logs")
            os.makedirs(self.log_dir)
        else:
            pass

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

        # create main dir if not exist
        self.__create_main_directory()

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
            json.dump(d, f, indent = 4)


class Evaluation():

    def __init__(self, params, filename, model, mode):
        self.params = params
        self.model_dir = params.model_directory
        self.mode = mode
        self.model = model
        self.model_input_shape = (1, params.img_shape, params.img_shape, 3)
        self.filename = filename
        self.image, self.label = self.__load_test_image()
        self.prediction = self.__predict_image()
        self.seg_cmap, self.seg_norm, self.bounds = self.color_mappings()
        self.jaccard = jaccard_score(self.label.flatten(), self.prediction.flatten(), average = None)

    def color_mappings(self):
        color_palett = np.array([[148., 158., 167.],
                                 [11., 151., 199.],
                                 [30., 122., 57.],
                                 [135., 191., 234.],
                                 [37., 111., 182.],
                                 [156., 99., 84.],
                                 [226., 148., 60.],
                                 [203., 54., 68.],
                                 [192., 194., 149.],
                                 [105., 194., 185.],
                                 [209., 227., 239.],
                                 [226., 233., 48.]])

        color_palett_norm = color_palett / 255  # (np.max(color_palett)-np.min(color_palett))
        custom_cmap = colors.ListedColormap(
            color_palett_norm
        )

        # set counts and norm
        array_bounds = np.arange(13) - 0.1
        bounds = array_bounds.tolist()

        norm = colors.BoundaryNorm(bounds, custom_cmap.N)

        return custom_cmap, norm, bounds

    def resize(self, im):
        desired_size = self.params.img_shape
        im = Image.fromarray(im)

        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, Image.NEAREST)
        # create a new image and paste the resized on it

        new_im = Image.new("L", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

        return np.array(new_im)

    def __load_test_image(self):
        # load samples
        im = Image.open(os.path.join(self.params.data_path, "hq_images", self.filename))
        lbl = Image.open(os.path.join(self.params.data_path, "hq_masks", self.filename))

        im = np.array(im)
        lbl = np.array(lbl)

        # resize samples
        im_resized = self.resize(im)

        # if image grey scale, make 3 channel
        if len(im_resized.shape) == 2:
            im_resized = np.stack((im_resized,) * 3, axis = -1)

        lbl_resized = self.resize(lbl)

        im_scaled = np.divide(im_resized, 255., dtype = np.float32)

        return im_scaled, lbl_resized.astype(int)

    def __predict_image(self):
        # get probability map
        pred = self.model.predict(self.image.reshape(self.model_input_shape))

        return np.argmax(pred, -1)[0, :, :].astype(int)

    def plot_record(self):
        seg_cmap, seg_norm, bounds = self.color_mappings()
        fig = plt.figure(figsize = (16, 4))

        gs = gridspec.GridSpec(nrows = 1,
                               ncols = 3,
                               figure = fig,
                               width_ratios = [1, 1, 1],
                               height_ratios = [1],
                               wspace = 0.3,
                               hspace = 0.3)

        # turn image to 3 channel
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.image)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("oct")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.label, cmap=seg_cmap, norm=seg_norm)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("ground truth")

        ax3 = fig.add_subplot(gs[0, 2])
        colorbar_im = ax3.imshow(self.prediction, cmap=seg_cmap, norm=seg_norm)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("prediction")

        # set colorbar ticks
        tick_loc_array = np.arange(13) + 0.5
        tick_loc_list = tick_loc_array.tolist()

        tick_list = np.arange(13).tolist()

        c_bar = plt.colorbar(colorbar_im, cmap=seg_cmap, norm=seg_norm, boundaries=bounds)

        # set ticks
        c_bar.set_ticks(tick_loc_list)
        c_bar.ax.set_yticklabels(tick_list)

        if not os.path.exists(os.path.join(self.model_dir, self.mode + "_predictions")):
            os.makedirs(os.path.join(self.model_dir, self.mode + "_predictions"))

        plt.savefig(os.path.join(self.model_dir, self.mode + "_predictions", self.filename))
        plt.close()


class TrainOps():
    def __init__(self, params):
        self.params = params

    def color_mappings(self):
        color_palett = np.array([[148., 158., 167.],
                                 [11., 151., 199.],
                                 [30., 122., 57.],
                                 [135., 191., 234.],
                                 [37., 111., 182.],
                                 [156., 99., 84.],
                                 [226., 148., 60.],
                                 [203., 54., 68.],
                                 [192., 194., 149.],
                                 [105., 194., 185.],
                                 [209., 227., 239.],
                                 [226., 233., 48.]])

        color_palett_norm = color_palett / 255  # (np.max(color_palett)-np.min(color_palett))
        custom_cmap = colors.ListedColormap(
            color_palett_norm
        )

        # set counts and norm
        array_bounds = np.arange(13) - 0.1
        bounds = array_bounds.tolist()

        norm = colors.BoundaryNorm(bounds, custom_cmap.N)

        return custom_cmap, norm, bounds

    def plot_examples(self, record, name):
        seg_cmap, seg_norm, bounds = self.color_mappings()
        fig = plt.figure(figsize = (16, 8))
        columns = 2
        rows = 1
        for i in range(1, columns * rows + 1):
            img = record[i - 1]
            fig.add_subplot(rows, columns, i)
            
            # 
            if i == 1:
                plt.imshow(img)
            if i == 2:
                plt.imshow(img, cmap=seg_cmap, norm=seg_norm)
                
            plt.savefig(self.params.model_directory + "/exmaple_{}.png".format(name))
        plt.close()

    def dice_loss(self, y_true, y_pred):
        num_labels = self.params.num_classes

        probabilities = tensorflow.keras.backend.reshape(y_pred, [-1, num_labels])
        y_true_flat = tensorflow.keras.backend.reshape(y_true, [-1])
    
        onehots_true = tensorflow.one_hot(tensorflow.cast(y_true_flat, tensorflow.int32), num_labels)
    
        numerator = tensorflow.reduce_sum(onehots_true * probabilities, axis=-1)
        denominator = tensorflow.reduce_sum(onehots_true + probabilities, axis=-1)
    
        loss = 1.0 - 2.0 * (numerator / denominator)
        return tensorflow.keras.backend.mean(loss)
    
    def gen_dice(self, y_true, y_pred, eps=1e-6):
        """both tensors are [b, h, w, classes] and y_pred is in logit form"""

        # [b, h, w, classes]
        pred_tensor = y_pred
        y_true_shape = tf.shape(y_true)

        # [b, h*w, classes]
        y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
        y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])

        # [b, classes]
        # count how many of each class are present in 
        # each image, if there are zero, then assign
        # them a fixed weight of eps
        #counts = tf.reduce_sum(y_true, axis=1)
        #weights = 1. / (counts ** 2)
        #weights = tf.where(tf.math.is_finite(weights), weights, eps)

        multed = tf.reduce_sum(y_true * y_pred, axis=1)
        summed = tf.reduce_sum(y_true + y_pred, axis=1)

        # [b]
        numerators = tf.reduce_sum( multed, axis=-1)
        denom = tf.reduce_sum(summed, axis=-1)
        dices = 1. - 2. * numerators / denom
        dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
        return tf.reduce_mean(dices)

    def lr_schedule(self, epoch):
        """Learning Rate Schedule
    
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = self.params.learning_rate

        if epoch > 850:
            lr *= 1e-3
        elif epoch > 800:
            lr *= 1e-2
        elif epoch > 200:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def callbacks_(self):
        '''callbacks'''
        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        checkpoint = ModelCheckpoint(filepath = self.params.model_directory + "/weights.hdf5",
                                     monitor = 'val_loss',
                                     save_best_only = True,
                                     verbose = 1,
                                     save_weights_only = True)

        tb = TensorBoard(log_dir = self.params.model_directory + "/tensorboard",
                         histogram_freq = 0,
                         write_graph = True,
                         write_images = True,
                         embeddings_layer_names = None,
                         embeddings_metadata = None)

        csv_logger = CSVLogger(filename = self.params.model_directory + '/history.csv',
                                                          append = True,
                                                          separator = ",")

        return [lr_scheduler, checkpoint, tb, csv_logger]

