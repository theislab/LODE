import os
from pprint import pprint
import pandas as pd
import numpy as np
from keras.engine.saving import load_model
from segmentation_models.metrics import iou_score
from sklearn.metrics import jaccard_score, classification_report

from feature_segmentation.generators.generator_utils.image_processing import read_resize
from feature_segmentation.utils.utils import Params, cast_params_types


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


def predict(model, img):
    """
    Parameters
    ----------
    model : keras model for segmentation
    img : array, numpy array with preprocessed image to predict on

    Returns
    -------
    integer label map from prediction and soft max scores for each class
    """

    # check so shape is 4 channel
    if len(img.shape) == 3:
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[-1]))

    # get probability map
    pred = model.predict(img)
    return np.argmax(pred, -1)[0, :, :].astype(int), pred


target_dict = {0: 'class 0', 1: 'class 1', 2: 'class 2',
               3: 'class 3', 4: 'class 4', 5: 'class 5',
               6: 'class 6', 7: 'class 7', 8: 'class 8',
               9: 'class 9', 10: 'class 10', 11: 'class 11',
               12: 'class 12', 13: 'class 13', 14: 'class 14',
               15: 'class 15'}


def present_targets(all_labels, all_predictions):
    """
    Parameters
    ----------
    all_labels : flatten list with all noted labels
    all_predictions : flatten list with a noten predictions

    Returns
    -------
    all the targets in dict to use for sklearn classification report
    """
    target_names = []
    labels_present = np.unique(np.unique(all_labels).tolist() + np.unique(all_predictions).tolist())
    for lp in labels_present:
        target_names.append(target_dict[lp])
    return target_names


def get_result_report(all_labels, all_predictions, model_directory):
    """
    Save the classification report and iou's for the model in its directory
    Parameters
    ----------
    all_labels : flatten list with all noted labels
    all_predictions : flatten list with a noten predictions
    model_directory : path to model

    Returns
    -------
    None
    """
    target_names = present_targets(all_labels, all_predictions)

    ious = jaccard_score(all_labels, all_predictions, average = None)
    clf_report = classification_report(all_labels, all_predictions, target_names = target_names, output_dict = 1)

    print(classification_report(all_labels, all_predictions, target_names = target_names))

    pprint(f"The class ious are: {ious}")
    pprint(f"The mIOU is {np.mean(ious)}")

    np.savetxt(model_directory + "/ious.txt", ious)
    pd.DataFrame(clf_report).to_csv(model_directory + "/clf_report.csv")


def load_test_config(model_path):
    """
    Parameters
    ----------
    model_path : path to specific model

    Returns
    -------
    keras model, test ids for the model, params object with model config
    """
    # load utils classes
    params = Params(os.path.join(model_path, "config.json"))

    # cast all numeric types to float and save to json
    cast_params_types(params, model_path)

    params = Params(os.path.join(model_path, "params.json"))

    # read test images from trained model
    test_ids = pd.read_csv(os.path.join(model_path, "test_ids.csv"))["0"].tolist()

    save_model_path = os.path.join(model_path, "weights.hdf5")

    model = load_model(save_model_path, custom_objects = {'iou_score': iou_score})
    return model, test_ids, params
