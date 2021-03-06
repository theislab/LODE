import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path

path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, str(path_variable))
sys.path.insert(0, str(path_variable.parent))

from models.networks.layers.attn_augconv import AttentionAugmentation2D

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from pydicom import read_file


from utils.utils import Params

SEGMENTED_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

target_dict = {0: 'class 0', 1: 'class 1', 2: 'class 2',
               3: 'class 3', 4: 'class 4', 5: 'class 5',
               6: 'class 6', 7: 'class 7', 8: 'class 8',
               9: 'class 9', 10: 'class 10', 11: 'class 11',
               12: 'class 12', 13: 'class 13', 14: 'class 14',
               15: 'class 15'}


def ensemble_vote(predictions):
    """
    Parameters
    ----------
    predictions : list of arrays with soft max predictions from all models on one image

    Returns
    -------
    ensemble prediction labels as an array
    """
    ensemble_prediction = np.mean(np.array(predictions), 0)
    return np.argmax(ensemble_prediction, -1)[0, :, :].astype(int)


def ensemble_uncertainty(predictions):
    """
    Parameters
    ----------
    predictions : list of arrays with soft max predictions from all models on one image

    Returns
    -------
    ensemble prediction labels as an array
    """
    prediction_array = np.array(predictions)

    ensemble_softmax = np.mean(prediction_array, 0)

    ensemble_prediction = np.argmax(ensemble_softmax, -1)[0, :, :].astype(int).flatten()
    ensemble_std = np.std(np.array(predictions), 0).reshape(-1, 16)

    uncertainty = []
    for k, idx in enumerate(ensemble_prediction):
        uncertainty.append(ensemble_std[k, idx])

    uq_map = np.array(uncertainty).reshape(256, 256)
    return uq_map


def ensemble_predict(ensemble_dict, img):
    """
    Parameters
    ----------
    ensemble_dict : dict of models
    img : array, numpy array with preprocessed image to predict on

    Returns
    --------
    integer label map from prediction and soft max scores for each class
    """
    # check so shape is 4 channel
    if len(img.shape) == 3:
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[-1]))

    # pre process (255. divide) as when training
    img = img / 255.

    model_segmentations = {}
    predictions = []
    for ensemble_model in ensemble_dict.keys():
        model = ensemble_dict[ensemble_model]["model"]

        # get probability map
        pred = model.predict(img)

        predictions.append(pred)
        model_segmentations[ensemble_model] = np.argmax(pred, -1)[0, :, :].astype(int)

    uq_map = ensemble_uncertainty(predictions)
    return model_segmentations, ensemble_vote(predictions), uq_map


def check_enseble_test_ids(ensemble_dict):
    """
    This  function asserts so the ensemble models have been trained on the same samples
    Parameters
    ----------
    ensemble_dict : dict with all enseble information

    Returns
    -------
    None
    """
    # check so test ids are same
    not_same_ids = []
    for k, model in enumerate(ensemble_dict.keys()):
        model_test_ids = ensemble_dict[model]["test_ids"]
        model_test_ids.sort()

        if k > 0:
            if model_test_ids == last_models_ids:
                not_same_ids.append(False)
            else:
                not_same_ids.append(True)

        last_models_ids = model_test_ids

        assert sum(not_same_ids) == 0, "models in ensemble not trained on the same ids, stop evaluation"


def predict_on_batch(model, img):
    """
    Parameters
    ----------
    model : tensorflow.keras model for segmentation
    img : array, numpy array with preprocessed image to predict on

    Returns
    -------
    integer label map from prediction and soft max scores for each class
    """

    # check so shape is 4 channel
    if len(img.shape) == 3:
        img = img.reshape((img.shape[0], img.shape[0], img.shape[1], img.shape[-1]))

    # pre process (255. divide) as when training
    img = img / 255.
    
    # get probability map
    pred = model.predict_on_batch(img)
    return np.argmax(pred, -1).astype(np.uint8), pred


def predict(model, img):
    """
    Parameters
    ----------
    model : tensorflow.keras model for segmentation
    img : array, numpy array with preprocessed image to predict on

    Returns
    -------
    integer label map from prediction and soft max scores for each class
    """

    # check so shape is 4 channel
    if len(img.shape) == 3:
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[-1]))

    # pre process (255. divide) as when training
    img = img / 255.

    # get probability map
    pred = model.predict(img)
    return np.argmax(pred, -1)[0, :, :].astype(int), pred


def load_test_config(model_path):
    """
    Parameters
    ----------
    model_path : path to specific model

    Returns
    -------
    tensorflow.keras model, test ids for the model, params object with model config
    """

    # load utils classes
    params = Params(os.path.join(model_path, "config.json"))

    # cast all numeric types to float and save to json
    cast_params_types(params, model_path)

    params = Params(os.path.join(model_path, "params.json"))

    # read test images from trained model
    test_ids = pd.read_csv(os.path.join(model_path, "test_ids.csv"))["0"].tolist()
    validation_ids = pd.read_csv(os.path.join(model_path, "validation_ids.csv"))["0"].tolist()
    train_ids = pd.read_csv(os.path.join(model_path, "train_ids.csv"))["0"].tolist()

    save_model_path = os.path.join(model_path, "weights.hdf5")
    model = load_model(save_model_path, custom_objects={'AttentionAugmentation2D': AttentionAugmentation2D})

    # model = load_model(save_model_path)
    return model, test_ids, validation_ids, train_ids, params


def get_ensemble_dict(ensemble_models, models_directory):
    """
    Parameters
    ----------
    ensemble_models : list of model names (strings)

    Returns
    -------
    dict, with each model and corresponding params and test ids
    """
    ensemble_dict = {}

    # get ensemble config dict
    for ensemble_model in ensemble_models:
        model_path = os.path.join(models_directory, ensemble_model)

        # load test configurations
        model, test_ids, validation_ids, train_ids, params = load_test_config(model_path)

        ensemble_dict[ensemble_model] = {"model": model, "test_ids": test_ids, "params": params}
    return ensemble_dict


def save_volume(segmentation, save_path, record_name):
    """
    Parameters
    ----------
    segmentation : segmented oct
    save_path : str; where to save

    Returns
    -------
    None
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path + f"/{record_name}.npy", segmentation)


def save_segmentation(segmentation, save_path, record_name):
    """
    Parameters
    ----------
    segmentation : segmented oct
    save_path : str; where to save

    Returns
    -------
    None
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + f"/{record_name}.npy", segmentation)


def save_embedding(embeding, save_path, record_name):
    """
    Parameters
    ----------
    embeding : embedded oct volume
    save_path : str; where to save

    Returns
    -------
    None
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path + f"/{record_name}.npy", embeding)


def embedd(model, img):
    """
    Parameters
    ----------
    model : tensorflow.keras model for segmentation
    img : array, numpy array with preprocessed image to predict on

    Returns
    -------
    flattened embedding vector from bottle neck of segmenter
    """

    # check so shape is 4 channel
    if len(img.shape) == 3:
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[-1]))

    # pre process (255. divide) as when training
    img = img / 255.

    return model.predict(img).flatten()


def segment_volume(oct_volume, ensemble_dict):
    """
    Parameters
    ----------
    save_path :
    oct_volume : array of shape (n, 256, 256, 3) with all resized oct form a dicom
    ensemble_dict : dict holding all models in ensemble
    save_volume: boolean; if volume should be save
    Returns
    -------
    array with segmented OCTs
    """
    segmented_octs = []
    for i in range(oct_volume.shape[0]):
        _, segmented_oct = ensemble_predict(ensemble_dict, oct_volume[i])
        segmented_octs.append(segmented_oct)

    return np.array(segmented_octs)


def initialize_volume_feature_dict():
    """
    Returns
    -------
    dict, with all features and id element.
    """
    return {"frame": [], 0: [], 1: [], 2: [], 3: [], 4: [],
            5: [], 6: [], 7: [], 8: [], 9: [], 10: [],
            11: [], 12: [], 13: [], 14: [], 15: []}


def oct_segmentation_to_vector(segmentation):
    """
    Parameters
    ----------
    segmentation : array, segmented oct
    feature_dict : dict with counts for each feature

    Returns
    -------
    feature dict with feature statistics for segmentation
    """
    feature_dict = {}

    # count features
    feature_counts = np.unique(segmentation, return_counts = True)

    # add do dict
    for feature in SEGMENTED_CLASSES:
        if int(feature) in feature_counts[0]:
            feature_dict[feature] = feature_counts[1][feature_counts[0].tolist().index(int(feature))]
        else:
            feature_dict[feature] = 0
    return feature_dict


def segmentation_to_vector(segmentation, feature_dict={}):
    """
    Parameters
    ----------
    segmentation : array, segmented oct
    feature_dict : dict with counts for each feature

    Returns
    -------
    feature dict with feature statistics for segmentation
    """

    # count features
    feature_counts = np.unique(segmentation, return_counts = True)

    # add do dict
    for feature in SEGMENTED_CLASSES:
        if int(feature) in feature_counts[0]:
            feature_dict[feature].append(feature_counts[1][feature_counts[0].tolist().index(int(feature))])
        else:
            feature_dict[feature].append(0)
    return feature_dict


def get_feature_dict(dicom_path, segmented_volume):
    """
    Parameters
    ----------
    dicom_path : str, path to dicom
    segmented_volume : array, segmented oct volume

    Returns
    -------
    dict with all information from oct
    """
    dc_header = read_file(dicom_path, stop_before_pixels = True)

    feature_dict = initialize_volume_feature_dict()

    feature_dict["patient_id"] = dc_header.PatientID.replace("ps:", "")
    feature_dict["study_date"] = dc_header.StudyDate
    feature_dict["laterality"] = dc_header.ImageLaterality
    feature_dict["dicom_path"] = dicom_path
    
    num_segmentations = segmented_volume.shape[0]
    for i in range(num_segmentations):
        feature_dict["frame"].append(i)
        feature_dict = segmentation_to_vector(segmented_volume[i], feature_dict)

    return feature_dict


def get_embedding_model(model_directory):
    """
    Parameters
    ----------
    model_directory : str; full path to model

    Returns
    -------
    tensorflow.keras model to use for embedding the octs
    """
    save_model_path = os.path.join(model_directory, "weights.hdf5")
    model = load_model(save_model_path)

    # set up inference model
    model_input = model.layers[0]
    model_embedding = model.layers[len(model.layers) // 2]

    # inference model
    embedding_model = Model(inputs = model_input.output, outputs = [model_embedding.output])
    return embedding_model


def save_predicted_detections(pred, save_path, id_):
    """
    Parameters
    ----------
    pred :
    save_path :

    Returns
    -------

    """

    NON_TISSUE_LABELS = [0, 12, 11, 14, 15]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    detections = []
    from skimage.measure import regionprops
    for region in regionprops(pred):
        if region.area >= 100 and (region.label not in NON_TISSUE_LABELS):
            minr, minc, maxr, maxc = region.bbox
            detections.append((region.label, 1, minr, minc, maxr, maxc))

    pd.DataFrame(detections).to_csv(os.path.join(save_path, id_), sep = " ", header = None, index = None)


def save_groundtruth_detections(lbl, save_path, id_):
    """
    Parameters
    ----------
    pred :
    save_path :

    Returns
    -------

    """

    NON_TISSUE_LABELS = [0, 12, 11, 14, 15]

    if len(lbl.shape) > 2:
        lbl = lbl[:, :, 0]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    detections = []
    from skimage.measure import regionprops
    for region in regionprops(lbl):
        if region.area >= 100 and (region.label not in NON_TISSUE_LABELS):
            minr, minc, maxr, maxc = region.bbox
            detections.append((region.label, minr, minc, maxr, maxc))

    pd.DataFrame(detections).to_csv(os.path.join(save_path, id_), sep = " ", header = None, index = None)
