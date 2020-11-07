import os
from feature_segmentation.evaluate.evaluation_utils import get_result_report, load_test_config, \
    ensemble_predict, check_enseble_test_ids
from feature_segmentation.generators.generator_utils.image_processing import read_resize
from feature_segmentation.segmentation_config import WORK_SPACE, TRAIN_DATA_PATH
from feature_segmentation.utils.plotting import plot_image_label_prediction, plot_uncertainty_heatmaps, \
    plot_uncertainty_statistics, plot_image

# select model to be evaluated
ensemble_dir = "/media/olle/Seagate/LODE/workspace/segmentation_ensembles/ensemble_stratified_camera_effect"
models_directory = os.path.join(ensemble_dir, "models")
ensemble_models = os.listdir(models_directory)  # ["56", "57"]

if not os.path.exists(ensemble_dir):
    os.makedirs(ensemble_dir, exist_ok=True)

ensemble_dict = {}

# get ensemble config dict
for ensemble_model in ensemble_models:
    model_path = os.path.join(models_directory, ensemble_model)

    # load test configurations
    model, test_ids, params = load_test_config(model_path)

    ensemble_dict[ensemble_model] = {"model": model, "test_ids": test_ids, "params": params}

# assert correct train test split across ensemble
check_enseble_test_ids(ensemble_dict)

all_predictions = []
all_labels = []
all_uq_maps = {}

image_iter = 0
for i in range(0, len(test_ids) - 1):
    img_path = os.path.join(TRAIN_DATA_PATH, "images", test_ids[i])
    label_path = os.path.join(TRAIN_DATA_PATH, "masks", test_ids[i])

    img, lbl = read_resize(img_path, label_path, (params.img_shape, params.img_shape))

    # run prediction for each ensemble model
    model_predictions, ensemble_prediction, uq_map = ensemble_predict(ensemble_dict, img)

    all_predictions.extend(ensemble_prediction.flatten().tolist())
    all_labels.extend(lbl.flatten().tolist())
    all_uq_maps[test_ids[i]] = uq_map

    # plot all images and their labels/predictions
    plot_image_label_prediction([img, lbl, ensemble_prediction], ensemble_dir, mode="test", filename=test_ids[i])

    for model in model_predictions.keys():
        prediction = model_predictions[model]
        plot_image_label_prediction([img, lbl, ensemble_prediction], ensemble_dir, mode=f"test_{model}",
                                    filename=test_ids[i])
        plot_image([lbl[:, :, 0]], ensemble_dir, mode=f"test_{model}",
                   filename=test_ids[i])

# plot all uq maps
plot_uncertainty_heatmaps(all_uq_maps, ensemble_dir)
plot_uncertainty_statistics(all_uq_maps, ensemble_dir)

# generate results files from all prediction and labels and save to model directory
get_result_report(all_labels, all_predictions, ensemble_dir)
