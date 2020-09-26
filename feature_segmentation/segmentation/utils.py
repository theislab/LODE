"""General utility functions"""
import json
import cv2
from PIL import Image
import numpy as np
import os
from pydicom import read_file
import keras.backend as K
from feature_segmentation.utils.im_processing import resize
from feature_segmentation.utils.utils import Params, Logging, TrainOps


def load_config(model_directory):
    """
    :param model_directory: path to model to load
    :type model_directory: str
    :return: params, logging, and trainops config
    :rtype: custom objects
    """
    params = Params(os.path.join(model_directory, "config.json"))
    params.model_directory = model_directory

    # cast data types to numeric
    params = params.dict
    for k in params.keys():
        try:
            int(params[k])
            params[k] = int(params[k])
        except ValueError:
            try:
                float(params[k])
                params[k] = float(params[k])
            except ValueError:
                pass

    with open(os.path.join(model_directory, 'params.json'), 'w') as json_file:
        json.dump(params, json_file)

    params = Params(os.path.join(model_directory, "params.json"))
    logging = Logging("./logs", params)
    trainops = TrainOps(params)

    params.is_training = False
    return params, logging, trainops


class EvalVolume():
    def __init__(self, params, path, model, mode, volume_save_path,
                 embedd_save_path, save_volume=False, save_embedding=False, n_scan=49):
        self.selected_bscans = None
        self.params = params
        self.model_dir = params.model_directory
        self.save_volume = save_volume
        self.mode = mode
        self.model = model
        self.n_scan = n_scan
        self.save_embedding = save_embedding
        self.embedd_path = embedd_save_path
        self.path = path
        self.embedd_save_path = embedd_save_path
        self.volume_save_path = volume_save_path
        self.filename = path.split("/")[-1]
        self.model_input_shape = (1, params.img_shape, params.img_shape, 3)
        self.dicom = read_file(self.path)
        self.pat_id = self.dicom.PatientID.replace("ps:", "")
        self.laterality = self.dicom.ImageLaterality
        self.study_date = self.dicom.StudyDate
        self.series_id = self.dicom.SeriesNumber
        self.record_id = f"{self.pat_id}_{self.study_date}_{self.laterality}_{self.series_id}"
        self.create_save_directories()
        self.image = self.load_volume()
        self.segmented_volume = self.segment_volume()
        self.feature_dict = {"id": [], "0": [], "1": [], "2": [], "3": [], "4": [],
                             "5": [], "6": [], "7": [], "8": [], "9": [], "10": [],
                             "11": [], "12": [], "13": [], "14": [], "15": []}

    def create_save_directories(self):
        if not os.path.exists(self.embedd_save_path):
            os.makedirs(self.embedd_save_path)
        if not os.path.exists(self.volume_save_path):
            os.makedirs(self.volume_save_path)

    def load_volume(self):
        vol = self.dicom.pixel_array
        resized_vol = np.zeros((49 // self.n_scan, 256, 256, 3))
        idx = np.round(np.linspace(0, vol.shape[0] - 1, vol.shape[0] // self.n_scan)).astype(int)

        self.selected_bscans = idx
        volume = vol[idx, :, :]
        for i in range(0, volume.shape[0]):
            im_resized = resize(volume[i, :, :], self.params.img_shape)

            # if image grey scale, make 3 channel
            if len(im_resized.shape) == 2:
                im_resized = np.stack((im_resized,) * 3, axis = -1)

            im_scaled = np.divide(im_resized, 255., dtype = np.float32)
            resized_vol[i, :, :, :] = im_scaled
        return resized_vol

    def __save_volume(self, prediction):
        np.save(os.path.join(self.volume_save_path, self.record_id + ".npy"), prediction)

    def __predict_image(self, img):
        pred, embedd = self.model.predict(img.reshape(self.model_input_shape))
        return np.argmax(pred, -1)[0, :, :].astype(int), embedd

    def segment_volume(self):
        predictions = np.zeros(shape = (49 // self.n_scan, 256, 256), dtype = np.uint8)
        embeddings = np.zeros(shape = (49 // self.n_scan, 8, 8, 512), dtype = np.float32)
        for i in range(self.image.shape[0]):
            predictions[i, :, :], embeddings[i, :, :, :] = self.__predict_image(self.image[i, :, :, :])

        if self.save_volume:
            self.__save_volume(predictions)

        if self.save_embedding:
            self.__save_embedding(embeddings)
        return predictions

    def __save_embedding(self, embeddings):
        np.save(os.path.join(self.embedd_path, self.record_id + ".npy"), np.array(embeddings))

    def feature_statistics(self, selected_bscans):
        file_name = f"{self.pat_id}_{self.study_date}_{self.laterality}"
        frame_ids = np.arange(0, 49, self.n_scan).tolist()
        for i in range(self.segmented_volume.shape[0]):
            map_ = self.segmented_volume[i, :, :]

            # count features
            map_counts = np.unique(map_, return_counts = True)

            # add do dict
            for k, feature in enumerate(self.feature_dict.keys()):
                if feature == 'id':
                    # feature id is the dicom file name and frame number
                    self.feature_dict[feature].append(file_name + "_{}".format(frame_ids[i]))
                else:
                    if int(feature) in map_counts[0]:
                        self.feature_dict[feature].append(map_counts[1][map_counts[0].tolist().index(int(feature))])
                    else:
                        self.feature_dict[feature].append(0)

        self.feature_dict["patient_id"] = self.pat_id
        self.feature_dict["study_date"] = self.study_date
        self.feature_dict["laterality"] = self.laterality
        self.feature_dict["frame"] = selected_bscans.tolist()
        self.feature_dict["dicom"] = self.path.split("/")[-1]
        return self.feature_dict


class EvalBScan():
    def __init__(self, params, path, model, mode, volume_save_path,
                 embedd_save_path, save_volume=False, save_embedding=False):
        self.params = params
        self.model_dir = params.model_directory
        self.save_volume = save_volume
        self.mode = mode
        self.model = model
        self.save_embedding = save_embedding
        self.embedd_save_path = embedd_save_path
        self.path = path
        self.volume_save_path = volume_save_path
        self.filename = path.split("/")[-1]
        self.model_input_shape = (1, params.img_shape, params.img_shape, 3)
        self.image = self.load_oct()
        self.segmented_oct = self.segment_oct()
        self.feature_dict = {"id": [], "0": [], "1": [], "2": [], "3": [], "4": [],
                             "5": [], "6": [], "7": [], "8": [], "9": [], "10": [],
                             "11": [], "12": [], "13": [], "14": [], "15": []}

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

    def load_oct(self):

        oct_ = cv2.imread(self.path)

        # extract record identifier
        self.record_id = self.path.split("/")[-1]

        # resize samples
        im_resized = self.resize(oct_)

        # if image grey scale, make 3 channel
        if len(im_resized.shape) == 2:
            im_resized = np.stack((im_resized,) * 3, axis = -1)

        im_scaled = np.divide(im_resized, 255., dtype = np.float32)
        resized_oct = im_scaled

        return resized_oct

    def __save_oct(self, prediction):
        np.save(os.path.join(self.volume_save_path, self.record_id + ".npy"), prediction)

    def __predict_image(self, img):
        # get probability map
        pred, embed = self.model.predict(img.reshape(self.model_input_shape))

        return np.argmax(pred, -1)[0, :, :].astype(int), embed

    def __get_embedding(self, img):
        ind = len(self.model.layers)
        get_tensor_values = K.function([self.model.layers[0].input],
                                       [self.model.layers[ind // 2].output])
        result = get_tensor_values([img.reshape(1, img.shape[0], img.shape[1], 3)])[0]
        return result

    def segment_oct(self):
        prediction, embedding = self.__predict_image(self.image.reshape(1, 256, 256, 3))

        if self.save_volume:
            self.__save_oct(prediction)

        if self.save_embedding:
            self.__save_embedding(embedding)
        return prediction

    def __save_embedding(self, embedding):
        # save volume of embeddings
        np.save(os.path.join(self.embedd_path, self.record_id + ".npy"), np.array(embedding))

    def feature_statistics(self):
        map_ = self.segmented_oct

        # count features
        map_counts = np.unique(map_, return_counts = True)

        # add do dict
        for k, feature in enumerate(self.feature_dict.keys()):
            if feature == 'id':
                self.feature_dict[feature].append(self.filename + "_{}".format(i))
            else:
                if int(feature) in map_counts[0]:
                    self.feature_dict[feature].append(map_counts[1][map_counts[0].tolist().index(int(feature))])
                else:
                    self.feature_dict[feature].append(0)

        return self.feature_dict
