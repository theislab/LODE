import os
import model as mt
from train_eval_ops import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.layers import Input
from params import params, gen_params
import pandas as pd
import matplotlib.pyplot as plt
from python_generator import DataGenerator

from DeepRT.thickness_segmentation.params import gen_params
from DeepRT.thickness_segmentation.params import params
from DeepRT.thickness_segmentation.train_eval_ops import dice_loss, dice_coeff


def return_data_fromcsv_files(params, dataset):
    train_file = os.path.join( params["data_dir"], "file_names_complete", "train_new_old_mapping.csv" )
    val_file = os.path.join( params["data_dir"], "file_names_complete", "validation_new_old_mapping.csv" )
    test_file = os.path.join( params["data_dir"], "file_names_complete", "test_new_old_mapping.csv" )

    train_files = pd.read_csv(train_file, index_col = False )
    val_files = pd.read_csv(val_file, index_col = False )
    test_files = pd.read_csv(test_file, index_col = False )

    if dataset == "topcon":
        train_files = train_files[train_files["old_id"].str.contains("p_")]
        val_files = val_files[val_files["old_id"].str.contains( "p_" )]
        test_files = test_files[test_files["old_id"].str.contains( "p_" )]
    if dataset == "spectralis":
        train_files = train_files[~train_files["old_id"].str.contains( "p_" )]
        val_files = val_files[~val_files["old_id"].str.contains( "p_" )]
        test_files = test_files[~test_files["old_id"].str.contains( "p_" )]

    partition = {"train": train_files["new_id"].tolist(),
                 "validation": val_files["new_id"].tolist(),
                 "test":test_files["new_id"].tolist()}
    return partition


def return_model(params):
    '''get model'''
    input_img = Input(params["img_shape"], name = 'img' )
    model = mt.get_unet(input_img,
                         n_filters = 12,
                         dropout = 0.5,
                         batchnorm = True,
                         training = False)
    return model

def load_record(id):
    # load example
    oct = cv2.imread( os.path.join( params["data_dir"],
                                    "all_images",
                                    str(id) + ".png" ) )

    label = cv2.imread( os.path.join( params["data_dir"],
                                      "all_labels",
                                      str(id) + ".png" ) )[:, :, 0]

    # resize samples
    oct_resized = cv2.resize( oct, (256, 256) ).reshape( 1, 256, 256, 3 )
    label_resized = np.array( Image.fromarray( label ).resize( (256, 256) ) ).reshape( (1, 256, 256, 1) )

    # scaling
    label_scaled = label_resized / 255.
    oct_scaled = np.divide( oct_resized, 255., dtype = np.float32 )
    # label_im = np.nan_to_num(label_im)
    oct_scaled = np.nan_to_num( oct_scaled )
    return oct_scaled, label_scaled

# get model
model = return_model(params)

'''train and save model'''
save_model_path = os.path.join(params["save_path"],
                               "weights.hdf5" )
'''Load models trained weights'''
model.load_weights(save_model_path, by_name = True, skip_mismatch = True )

sets = ["test"]
devices = ["topcon", "spectralis"]

for device in devices:
    # get file names for generator
    partition = return_data_fromcsv_files(params, dataset = device)
    for set in sets:
        result = {}
        result[set+"_"+device] = []
        for id in partition[set]:
            # load record
            oct, label = load_record(id)

            # Train model on dataset
            oct_prediction = model.predict(oct)

            # binarize image
            oct_prediction[oct_prediction < 0.5] = 0
            oct_prediction[oct_prediction > 0.5] = 1

            cv2.imwrite( "./data/all_predictions/" + str( id ) + ".png", np.stack( (oct_prediction[0, :, :, 0],) * 3, axis = -1 )*255 )

            # get metric
            iou = compute_iou(label.flatten(),oct_prediction.flatten())

            result[set+"_"+device].append(iou)
            print(iou)

        ids = pd.DataFrame(partition[set])
        ids["iou"] = pd.DataFrame.from_dict(result)
        ids.to_csv(set+"_id-mapping_"+device+".csv",index=False)

