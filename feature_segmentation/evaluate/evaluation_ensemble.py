import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report
import numpy as np
import pandas as pd

from feature_segmentation.evaluate.evaluation_utils import get_result_report, load_test_config, \
    ensemble_predict, check_enseble_test_ids
from feature_segmentation.generators.generator_2d import label_mapping
from feature_segmentation.generators.generator_utils.image_processing import read_resize
from feature_segmentation.segmentation_config import TRAIN_DATA_PATH
from feature_segmentation.utils.plotting import plot_image_label_prediction, plot_uncertainty_heatmaps, \
    plot_uncertainty_statistics, plot_image, plot_label, plot_predictions

tf.compat.v1.disable_eager_execution()

configured_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15]
metrics = ["iou", "precision", "recall", "f1_score", "support"]

result_metrics = ['record']

for cl in configured_labels:
    for m in metrics:
        result_metrics.append(f"{cl}_{m}")

# select model to be evaluated
ensemble_dir = "/home/olle/PycharmProjects/LODE/feature_segmentation/trained_model/ensemble_curated"
ensemble_models = ["61", "60", "59", "58", "57", "56", "55", "54", "53", "52"]


result_pd = pd.DataFrame(columns = result_metrics)

if not os.path.exists(ensemble_dir):
    os.makedirs(ensemble_dir, exist_ok = True)

ensemble_dict = {}

# get ensemble config dict
for ensemble_model in ensemble_models:
    model_path = os.path.join(ensemble_dir, ensemble_model)

    # load test configurations
    model, test_ids, validation_ids, train_ids, params = load_test_config(model_path, tta = False)

    ensemble_dict[ensemble_model] = {"model": model, "test_ids": test_ids, "params": params}

# assert correct train test split across ensemble
check_enseble_test_ids(ensemble_dict)

all_predictions = []
all_labels = []
all_uq_maps = {}

test_ids = test_ids
mode = "test"
image_iter = 0

for i, id_ in enumerate(test_ids):
    img_path = os.path.join(TRAIN_DATA_PATH, "images", id_)
    label_path = os.path.join(TRAIN_DATA_PATH, "masks", id_)

    img, lbl = read_resize(img_path, label_path, (params.img_shape, params.img_shape))

    # map artifacts and sereous ped to zero or fv-ped
    lbl = label_mapping(lbl)

    # run prediction for each ensemble model
    model_predictions, ensemble_prediction, uq_map = ensemble_predict(ensemble_dict, img)

    prediction_list = ensemble_prediction.flatten().tolist()
    labels_list = lbl.flatten().tolist()

    all_predictions.extend(prediction_list)
    all_labels.extend(labels_list)
    all_uq_maps[id_] = uq_map

    present_labels = list(set(prediction_list).union(labels_list))

    ious = jaccard_score(labels_list, prediction_list, average = None, labels = configured_labels, zero_division=1)
    iou_dict = dict(zip(configured_labels, ious))

    clf_report = classification_report(labels_list, prediction_list, labels = configured_labels,
                                       output_dict = 1, zero_division=1)

    result_dict = {"record": id_}

    for present_label in configured_labels:
        if present_label in present_labels:
            result_dict[f"{present_label}_iou"] = iou_dict[present_label]
        else:
            result_dict[f"{present_label}_iou"] = np.nan

    for c_label in configured_labels:
        if str(c_label) in clf_report.keys():
            class_clf_report = clf_report[str(c_label)]
            for subkey in class_clf_report.keys():
                result_dict[f"{c_label}_{subkey}"] = class_clf_report[subkey]
        else:
            for subkey in clf_report['0'].keys():
                result_dict[f"{c_label}_{subkey}"] = np.nan

    result_pd = result_pd.append(pd.DataFrame(result_dict, index = [i]))

    # plot all images and their labels/predictions
    plot_image_label_prediction([img, lbl, ensemble_prediction], ensemble_dir, mode = mode, filename = id_)

    # plot all images and their labels/predictions
    plot_image(image = img, model_dir = ensemble_dir, mode = mode, filename = id_)
    plot_label(lbl, model_dir = ensemble_dir, mode = mode, filename = id_)
    plot_predictions([ensemble_prediction], model_dir = ensemble_dir, mode = mode, filename = id_)

    if not os.path.exists(os.path.join(ensemble_dir, mode + "_predictions")):
        os.makedirs(os.path.join(ensemble_dir, mode + "_predictions"), exist_ok = True)

    if not os.path.exists(os.path.join(ensemble_dir, mode + "_labels")):
        os.makedirs(os.path.join(ensemble_dir, mode + "_labels"), exist_ok = True)

    np.save(os.path.join(ensemble_dir, mode + "_predictions", id_.replace(".png", ".npy")), ensemble_prediction)
    np.save(os.path.join(ensemble_dir, mode + "_labels", id_.replace(".png", ".npy")), lbl)

    for model in model_predictions.keys():
        prediction = model_predictions[model]
        plot_image_label_prediction([img, lbl, ensemble_prediction], ensemble_dir, mode = f"{mode}_{model}",
                                    filename = id_)

# plot all uq maps
plot_uncertainty_heatmaps(all_uq_maps, ensemble_dir)
plot_uncertainty_statistics(all_uq_maps, ensemble_dir)

# generate results files from all prediction and labels and save to model directory
get_result_report(all_labels, all_predictions, ensemble_dir)

ious = ious_model = jaccard_score(all_labels, all_predictions, average = None, labels = np.unique(all_labels))

cm = confusion_matrix(all_labels, all_predictions)

np.savetxt(f"{ensemble_dir}/{mode}_cm.txt", cm)
np.savetxt(f"{ensemble_dir}/predictions.txt", all_predictions)
np.savetxt(f"{ensemble_dir}/labels.txt", all_labels)
np.savetxt(f"{ensemble_dir}/ious.txt", ious)

result_pd.to_csv(f"{ensemble_dir}/record_performance.csv")
