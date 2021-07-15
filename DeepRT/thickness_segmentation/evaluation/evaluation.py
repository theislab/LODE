import os
import pandas as pd
from tensorflow.keras.optimizers import *
import model as mt
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
from train_eval_ops import *
import numpy as np
import tensorflow as tf
import loading_numpy_functions as lnf
from params import params
import cv2

#save_pred_path= "/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/data/clinic_data/\
#topcon_predictions_spectralis_topcon_trained2"

save_pred_path= "/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/\
evaluation/statistics/trained_device_I_II_eval_device_II/train"


def return_model(params,model_iter,filters):
    '''get model'''
    input_img = Input(params["img_shape"], name='img')
    if model_iter == 1:
        model = mt.get_bunet(input_img, n_filters=4*filters, dropout=params["drop_out"], batchnorm=True, training=True)

    if model_iter == 2:
        model = mt.get_shallow_bunet(input_img, n_filters=4*filters, dropout=params["drop_out"], batchnorm=True, training=True)

    if model_iter == 3:
        model = mt.get_very_shallow_bunet(input_img, n_filters=4*filters, dropout=params["drop_out"], batchnorm=True, training=True)

    return model

def iou(predictions, labels):
    intersection = np.logical_and(predictions, labels)
    union = np.logical_or(predictions, labels)
    return np.sum(intersection) / np.sum(union).astype(float)

def main(input_shape, verbose, dr, lr,shape,model_it,num_filters,bf,cf):
    '''create save dir if does not exist'''
    try:
        os.stat(params["save_path"])
    except:
        os.makedirs(params["save_path"])

    '''load data files''' #
    files_dir = os.path.join(params["data_dir"], "file_names_topcon","train.csv")
    test_ids = pd.read_csv(files_dir).old_id.values

    num_test_examples = test_ids.shape[0]

    save_model_path = os.path.join(params["save_path"],"weights.hdf5")
    if not os.path.exists( params["save_path"]):
        os.makedirs( params["save_path"])

    gen_params['dim'] = (params["img_shape"][0], params["img_shape"][1])
    #bayesian parameters assigment

    # set image shape param
    dim = gen_params['dim'][0]
    # select one of the three models
    model_iter = max(int(model_it),1)
    # select one of the five filter levels
    filters = max(int(num_filters)*1,1)
    params["img_shape"] = (dim,dim,3)
    # set params for optimization
    params["drop_out"] = dr
    # learning rate
    params["learning_rate"] = lr
    # brightness rate
    gen_params["brightness_factor"] = bf
    # contrast rate
    gen_params["contrast_factor"] = cf

    opt = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #get model
    model = return_model(params,model_iter,filters)
    '''Compile model'''
    model.compile(optimizer=opt, loss=dice_loss)
    model.summary()

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"],"weights.hdf5")
    cp = tf.tensorflow.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor="val_loss",
                                            save_best_only=True, verbose=1,save_weights_only=True)

    if params["continuing_training"] == True:
        '''Load models trained weights'''
        model.load_weights(save_model_path,by_name=True, skip_mismatch=True)


    test_loss = []
    test_iou = []
    for i in range(0,num_test_examples):
        im_name = test_ids[i]
        im_batch_val, labels_batch_val = lnf.eval_lod(params["data_dir"], im_name)

        #get predction on loss
        prediction = model.predict(im_batch_val)
        loss = model.evaluate(x=im_batch_val, y=labels_batch_val)

        #set classes
        prediction[prediction < 0.5] = 0
        prediction[prediction > 0.5] = 1

        #flatten records
        labels_flatten = labels_batch_val.flatten()
        predition_flatten = prediction.flatten()

        #save prediction
        save_pred = np.stack((prediction[0, :, :, 0],) * 3, axis=-1) * 255
        cv2.imwrite(os.path.join(save_pred_path,str(im_name)+".png"),save_pred)

        #append scores to logging lists
        test_loss.append(loss)


    print("The mean iou and loss is: {}, {},respective std's are: {},{}".format(np.mean(test_iou), np.mean(test_loss),
                                                    np.std(test_iou), np.std(test_loss)))
    #save stats in csv file
    segmentation_eval_pd = pd.DataFrame([test_ids, test_iou,test_loss]).T
    segmentation_eval_pd.to_csv(os.path.join(save_pred_path,"eval_stats.csv"))


verbose = 1
input_shape = (256,256,3)
# Bounded region of parameter space
pbounds = {'bf': 0.9916641411079563,
           'cf': 0.987157583671153,
           'model_it': 0.2121808781249,
           'shape': 3.0523303142525506,
           'lr': 0.0025980098111566333,
           'num_filters': 3.2290169061135647,
           'dr': 0.1784720139932052}


main(input_shape, verbose, pbounds['dr'], pbounds['lr'],pbounds['shape'],
     pbounds['model_it'],pbounds['num_filters'],pbounds['bf'],pbounds['cf'])