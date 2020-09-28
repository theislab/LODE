import os
import pandas as pd
from keras.engine.saving import load_model
from segmentation_models.metrics import iou_score
import time
from keras import Model
import matplotlib
from tqdm import tqdm
from segmentation.utils import EvalBScan, load_config
from deep_clustering.config import SEG_MODEL_DIR, VOL_SAVE_PATH, EMBEDD_SAVE_PATH

matplotlib.use('agg')

if __name__ == "__main__":

    # select model to be evaluated
    model_directory = SEG_MODEL_DIR
    params, logging, trainops = load_config(model_directory)

    test_ids = os.listdir("./example_octs")

    # copy remaining ids
    remaining_ids = test_ids.copy()

    save_model_path = os.path.join(params.model_directory, "weights.hdf5")
    model = load_model(save_model_path, custom_objects = {'iou_score': iou_score})

    # set up inference model
    model_input = model.layers[0]
    model_output = model.layers[-1]
    model_embedding = model.layers[len(model.layers) // 2]

    # inference model
    inference_model = Model(inputs = model_input.output, outputs = [model_output.output, model_embedding.output])

    feature_statistics = pd.DataFrame(
        columns = {'id', 'patient_id', 'laterality', 'study_date', 'frame', 'dicom', '0', '1', '2', '3',
                   '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'})

    # set save paths
    feature_save_path = f"segmentation/feature_tables/feature_statistics.csv"
    progress_save_path = f"segmentation/to_be_completed.csv"

    image_iter = 0
    start = time.time()
    print("Number of files to segment are: ", len(test_ids))
    for i in tqdm(range(0, len(test_ids))):
        evaluation = EvalBScan(params = params,
                               path = test_ids[i],
                               model = inference_model,
                               mode = "test",
                               volume_save_path = VOL_SAVE_PATH,
                               embedd_save_path = EMBEDD_SAVE_PATH,
                               save_volume = True,
                               save_embedding = True)

        # get record statistics
        stat = evaluation.feature_statistics(evaluation.selected_bscans)

        # make into data frame
        feature_statistics = feature_statistics.append(pd.DataFrame(stat), sort = True)

        # remove from list
        remaining_ids.remove(test_ids[i])

        # progress
        if i % 1 == 0:
            feature_statistics.to_csv(feature_save_path)
            pd.DataFrame(remaining_ids).to_csv(progress_save_path)
