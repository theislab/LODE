import os
from feature_segmentation.evaluate.evaluation_utils import predict, get_result_report, load_test_config, \
    save_predicted_detections, save_groundtruth_detections
from feature_segmentation.generators.generator_utils.image_processing import read_resize
from feature_segmentation.segmentation_config import WORK_SPACE, TRAIN_DATA_PATH
from feature_segmentation.utils.plotting import plot_image_label_prediction

# select model to be evaluated
models_directory = os.path.join(WORK_SPACE, "models")
model_name = "49"

model_path = "/home/olle/PycharmProjects/LODE/feature_segmentation/trained_logs/0" # os.path.join(models_directory, model_name)

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

    # save object detection metrics
    save_predicted_detections(prediction, os.path.join(model_path, "object_detection/detections"),
                              test_ids[i].replace(".png", ".txt"))
    save_groundtruth_detections(lbl, os.path.join(model_path, "object_detection/groundtruths"),
                                test_ids[i].replace(".png", ".txt"))

    # plot all images and their labels/predictions
    plot_image_label_prediction([img, lbl, prediction], model_path, mode = "test", filename = test_ids[i])

# generate results files from all prediction and labels and save to model directory
get_result_report(all_labels, all_predictions, model_path)
