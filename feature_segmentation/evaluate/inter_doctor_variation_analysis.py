import os
import numpy as np
import glob
import cv2
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, jaccard_score, classification_report

# define all paths
from feature_segmentation.evaluate.evaluation_utils import present_targets
from feature_segmentation.generators.generator_2d import label_mapping

proj_dir = "/home/olle/PycharmProjects/LODE/feature_segmentation"
result_path = "trained_model/inter_doctor_variation"

n_records = 30

# saved in png format
j = "johannes_masks"
b = "ben_masks"
m = "michael_masks"
c = "concensus_masks"

# saved in numpy format
model = "model_masks"

# define annotators
annotators = ["j", "b", "m", "model", "concensus"]

records_ids = os.listdir("/home/olle/PycharmProjects/LODE/feature_segmentation/trained_model/inter_doctor_variation/concensus_masks")[0:n_records]

# load all segmentations
segmentation_dict = {}

segmentation_dict["j"] = [cv2.imread(os.path.join(proj_dir, result_path, j, id), 0).flatten() for id in records_ids]
segmentation_dict["j"] = np.concatenate(segmentation_dict["j"]).tolist()

segmentation_dict["b"] = [cv2.imread(os.path.join(proj_dir, result_path, b, id), 0).flatten() for id in records_ids]
segmentation_dict["b"] = np.concatenate(segmentation_dict["b"]).tolist()

segmentation_dict["m"] = [cv2.imread(os.path.join(proj_dir, result_path, m, id), 0).flatten() for id in records_ids]
segmentation_dict["m"] = np.concatenate(segmentation_dict["m"]).tolist()

segmentation_dict["model"] = [label_mapping(cv2.resize(np.load(os.path.join(proj_dir,
                                                                            result_path,
                                                                            model,
                                                                            id.replace("png", "npy"))).astype(np.uint8),
                                                       (512, 496),
                                                       cv2.INTER_NEAREST)).flatten() for id in records_ids]

segmentation_dict["concensus"] = [cv2.imread(os.path.join(proj_dir, result_path, c, id), 0).flatten() for id in records_ids]
segmentation_dict["concensus"] = np.concatenate(segmentation_dict["concensus"]).tolist()

segmentation_dict["model"] = np.concatenate(segmentation_dict["model"]).tolist()

# map artifact and sereous ped
for annotator in annotators:
    segmentation_dict[annotator] = label_mapping(segmentation_dict[annotator])


ious_model = jaccard_score(segmentation_dict["concensus"], segmentation_dict["model"], average = None, labels = np.unique(segmentation_dict["concensus"]))
ious_ben = jaccard_score(segmentation_dict["concensus"], segmentation_dict["b"], average = None, labels = np.unique(segmentation_dict["concensus"]))
ious_johannes = jaccard_score(segmentation_dict["concensus"], segmentation_dict["j"], average = None, labels = np.unique(segmentation_dict["concensus"]))
ious_michael = jaccard_score(segmentation_dict["concensus"], segmentation_dict["m"], average = None, labels = np.unique(segmentation_dict["concensus"]))

f1_score_model = f1_score(segmentation_dict["concensus"], segmentation_dict["model"], average = None, labels = np.unique(segmentation_dict["concensus"]))
f1_score_ben = f1_score(segmentation_dict["concensus"], segmentation_dict["b"], average = None, labels = np.unique(segmentation_dict["concensus"]))
f1_score_johannes = f1_score(segmentation_dict["concensus"], segmentation_dict["j"], average = None, labels = np.unique(segmentation_dict["concensus"]))
f1_score_michael = f1_score(segmentation_dict["concensus"], segmentation_dict["m"], average = None, labels = np.unique(segmentation_dict["concensus"]))


def plot_idv_metric(model_, ben_, johannes_, michael_, concensus_labels, metric="iou"):
    model_pd = pd.DataFrame(model_)
    model_pd["evaluator"] = "model"
    model_pd["labels"] = np.unique(concensus_labels).tolist()

    ben_pd = pd.DataFrame(ben_)
    ben_pd["evaluator"] = "ben"
    ben_pd["labels"] = np.unique(concensus_labels).tolist()

    johannes_pd = pd.DataFrame(johannes_)
    johannes_pd["evaluator"] = "johannes"
    johannes_pd["labels"] = np.unique(concensus_labels).tolist()

    michael_pd = pd.DataFrame(michael_)
    michael_pd["evaluator"] = "michael"
    michael_pd["labels"] = np.unique(concensus_labels).tolist()

    result_pd = model_pd.append(ben_pd).append(johannes_pd).append(michael_pd)
    result_pd.rename(columns = {0:metric}, inplace = True)

    result_pd.to_csv(os.path.join(proj_dir, result_path, f"idv_{metric}_comparison.csv"))

    import seaborn as sns
    ax = sns.barplot(x="labels", y=metric,  hue="evaluator", data=result_pd)
    plt.savefig(os.path.join(proj_dir, result_path, f"idv_{metric}_comparison.png"))
    plt.close()


plot_idv_metric(ious_model, ious_ben, ious_johannes, ious_michael, segmentation_dict["concensus"], metric="iou")
plot_idv_metric(f1_score_model, f1_score_ben, f1_score_johannes, f1_score_michael, segmentation_dict["concensus"],
                metric="f1_score")