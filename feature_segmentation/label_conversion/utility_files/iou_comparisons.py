import pandas as pd
import cv2
import os
import glob
import numpy as np
from sklearn.metrics import jaccard_score

# set data path
data_path = "/home/olle/PycharmProjects/feature_segmentation-master/data/label_conversion"

# set data iteration
choroid = 'without_choroidHQ'
iteration = 'johannes_6'
sub_iteration = 'iteration_3_2'

# get paths for each annotator
ben_paths = glob.glob(os.path.join(data_path, iteration, 'bens_masks/*'))
johannes_paths = glob.glob(os.path.join(data_path, iteration, 'masks/*'))

# michael_paths = glob.glob(os.path.join(data_path, "without_choroidHQ/iteration3/iteration3_2/iteration_3_2_michael/data_dataset_voc/masks/*"))

# get record names
records = os.listdir(os.path.join(data_path, iteration, 'masks'))


def get_user_iou(ground_truths, prediction_1):
    # considering one annotator, calculate per class iou w.r.t to other annotators
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    mean_scores = []
    records = []
    for i, ground_truth in enumerate(ground_truths, 0):
        # print(ground_truth.split("/")[-1], prediction_1[i].split("/")[-1]) #, prediction_2[i].split("/")[-1])

        record = ground_truth.split("/")[-1]
        gt = cv2.imread(ground_truth).flatten()
        p1 = cv2.imread(prediction_1[i]).flatten()
        #p2 = cv2.imread(prediction_2[i]).flatten()

        s1 = jaccard_score(gt, p1, average = None, labels = labels)

        ground_truths_labelled = np.unique(gt).tolist()

        for lbl in labels:
            if lbl in ground_truths_labelled:
                continue
            else:
                s1[lbl] = np.nan

        mean_scores.append(s1)
        records.append(record)
    return pd.DataFrame(mean_scores), records


ben_iou, records_names = get_user_iou(ben_paths, johannes_paths)
johannes_iou, records_names = get_user_iou(johannes_paths, ben_paths)

# johannes_iou, records_names = get_user_iou(johannes_paths, ben_paths, michael_paths)
# michael_iou, records_names = get_user_iou(michael_paths, ben_paths)#,johannes_paths

# average per class iou score over each class
overall_means = pd.concat([ben_iou, johannes_iou]).groupby(level = 0).mean()
overall_means["records"] = records_names

jiou = johannes_iou.mean(axis=0)
biou = ben_iou.mean(axis=0)

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
overall_means[labels] = overall_means[labels].replace({0: np.nan})
overall_means[labels] = overall_means[labels].round(decimals=2)

# drop columns with high agreement
overall_means = overall_means.drop(columns=[0, 2, 11, 12])

# get a per record statistic
overall_means['record_iou'] = overall_means[[1, 3, 4, 5, 6, 7, 8, 9, 10]].mean(axis=1)

# sort values
overall_means = overall_means.sort_values(by='record_iou')

# save list with average maximum disagreement across annotators for each record
overall_means.to_csv(os.path.join(data_path, "without_choroidHQ/iteration3/iteration3_2", "iou_stats.csv"))