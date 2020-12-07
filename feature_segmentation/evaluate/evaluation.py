import os
from feature_segmentation.evaluate.evaluation_utils import predict, get_result_report, load_test_config
from feature_segmentation.generators.generator_utils.image_processing import read_resize
from feature_segmentation.segmentation_config import WORK_SPACE, TRAIN_DATA_PATH
from feature_segmentation.utils.plotting import plot_image_label_prediction

# select model to be evaluated
models_directory = os.path.join(WORK_SPACE, "models")
model_name = "49"

model_path = "/home/olle/PycharmProjects/LODE/workspace/ensemble_results/56" # os.path.join(models_directory, model_name)

# load test configurations
model, test_ids, params = load_test_config(model_path)

all_predictions = []
all_labels = []

image_iter = 0
for i in range(0, len(test_ids) - 1):
    img_path = os.path.join(TRAIN_DATA_PATH, "images", test_ids[i])
    label_path = os.path.join(TRAIN_DATA_PATH, "masks", test_ids[i])

    img, lbl = read_resize(img_path, label_path, (params.img_shape, params.img_shape))

    prediction, softmax_prediction = predict(model, img)

    all_predictions.extend(prediction.flatten().tolist())
    all_labels.extend(lbl.flatten().tolist())

    # plot all images and their labels/predictions
    plot_image_label_prediction([img, lbl, prediction], model_path, mode = "test", filename = test_ids[i])

# generate results files from all prediction and labels and save to model directory
get_result_report(all_labels, all_predictions, model_path)
