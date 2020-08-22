import numpy as np
import pandas as pd
from params import *
from keras.layers import Input
import model_test as mt
from train_eval_ops import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
#predictive distribution - combination of epestemic and aleatoric uncertatinty
def predictive_uncertainty(y_pred_mc):
    '''
    :param y_pred_mc: number of pixels x number of classes x  number of MC samples
    :return:
    '''
    #FOR EACH CLASS, GET mc AVERGARE OF PROB. THEN MULTIPLY REG WITH LOG OF AVERAGE AND SUM OVER CLASSES
    return -np.sum(np.mean(y_pred_mc,axis=2)*np.log(np.mean(y_pred_mc,axis=2)),axis=1)

#mutual information - epistemic uncertainty
def predictive_entropy(y_pred):
    '''
    :param y_pred: y_pred_mc:number of pixels x number of classes x  number of MC samples
    :return:
    '''
    return np.sum(y_pred * np.log(y_pred))

def instantiate_bunet(params,adam):
    '''
    :param params: params stating config info
    :param opt: an optimizer for the network
    :return: model object for prediction
    '''

    '''get model'''
    input_img = Input(params["img_shape"], name='img')
    model = mt.get_bunet(input_img, n_filters=16, dropout=0.5, batchnorm=True, training=True)

    adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    '''Compile model'''
    model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=[iou])

    model.summary()

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"], "weights.hdf5")
    model.load_weights(save_model_path)

    return model


def get_input(path):
    im = cv2.imread(os.path.join(path),0)
    im = cv2.resize(im,(params["img_shape"][0],params["img_shape"][1]),interpolation=cv2.INTER_NEAREST)
    return im

def get_output(path):
    lbl = cv2.imread(os.path.join(path),0)
    lbl = cv2.resize(lbl, (params["img_shape"][0], params["img_shape"][1]),interpolation=cv2.INTER_NEAREST)
    return lbl

def pre_process(train_im, label_im):
    # scaling
    train_im = np.divide(train_im, 255., dtype=np.float32)
    # set all nans to zero
    return(train_im.reshape(params["img_shape"][0],params["img_shape"][1],1),
           label_im.reshape(params["img_shape"][0],params["img_shape"][1],1))

def evaluation_load(im_path, lbl_path):
    input = get_input(im_path)
    output = get_output(lbl_path)
    input_, output_ = pre_process(input, output)
    return(input_.reshape(1,params["img_shape"][0],params["img_shape"][1],1),
           output_.reshape(1,params["img_shape"][0],params["img_shape"][1],1))

def get_image_and_label(gen_params,i):
    im_path = os.path.join(gen_params['image_path'], "images",i.replace(".png",".jpg"))
    lbl_path = os.path.join(gen_params['label_path'], "labels",i)
    im , lbl = evaluation_load(im_path, lbl_path)
    return(im,lbl)

ids_train = pd.read_csv(os.path.join(params["data_dir"], "train_records.csv"))
ids_val = pd.read_csv(os.path.join(params["data_dir"], "validation_records.csv"))
ids_test = pd.read_csv(os.path.join(params["data_dir"], "test_records.csv"))

#make to list
ids_train = ids_train.values.flatten().tolist()
ids_val = ids_val.values.flatten().tolist()


num_training_examples = len(ids_train)
num_val_examples = len(ids_val)

print("Number of training examples: {}".format(num_training_examples))
print("Number of validation examples: {}".format(num_val_examples))

partition = {'train': [ids_train[1]], 'validation': ids_val}

#init model
model = instantiate_bunet(params,adam)

for k,i in enumerate(partition["train"]):

    #load image
    im , lbl = get_image_and_label(gen_params,i)
    #predict
    pred = model.predict(im)
    pred_ = pred.reshape(-1, params["number_of_classes"])
    pred_mask = np.argmax(pred_, axis=1).reshape(256, 256)

    eval = model.evaluate(im,lbl)

    #if save_pred:
    #    np.save(os.path.join(params["save_predictions_dir"],i + ".npy"), pred[0,:,:,0])

    #general performance metricy