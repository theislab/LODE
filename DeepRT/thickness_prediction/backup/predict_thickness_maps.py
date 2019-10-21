from train_eval_ops import *
import tensorflow as tf
from keras.optimizers import adam
import model as mt
import os
import cv2
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from params import *
from python_generator import DataGenerator
from train_eval_ops import *
from keras.layers import Input
import sys
import matplotlib.pyplot as plt
import pandas as pd
import resnet as re

def crop_image(img,cond, tol=0):
    # img is image data
    # tol  is tolerance
    mask = cond>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

tm_dir = "/media/olle/Seagate/thickness_map_prediction/thickness_maps"
f_dir = "/media/olle/Seagate/thickness_map_prediction/fundus"

files_ = [i.replace(".npy","") for i in os.listdir(tm_dir)]

#read model
res_output,img_input = re.ResNet50(params["img_shape"], 1)
outputs = mt.decoder(res_output, n_filters=16, dropout=0.05, batchnorm=True)
model = Model(inputs=img_input, outputs=[outputs])
'''Compile model'''
adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss=custom_mae, metrics=[custom_mae,percentual_deviance])
model.summary()

'''train and save model'''
save_model_path = os.path.join("/home/olle/PycharmProjects/thickness_prediction/thickness_model_weights",
                               "weights.hdf5")

cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                        monitor='val_percentual_deviance',
                                        save_best_only=True, verbose=1,
                                        save_weights_only=True)

learning_rate_reduction = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1)

'''Load models trained weights'''
model.load_weights(save_model_path)

for i in range(0,1000):
    print(files_[i])
    #load samples
    im = cv2.imread(os.path.join(f_dir, files_[i]+".png"))
    lbl = np.load(os.path.join(tm_dir, files_[i]+".npy"))

    if im != []:

        # convert to three channel
        if im.shape[-1] != 3:
            im = np.stack((im,) * 3, axis=-1)

        c_im = crop_image(im, lbl, tol=0)
        c_lbl = crop_image(lbl, lbl, tol=0)


        #resize samples
        im_resized = cv2.resize(c_im, (params["img_shape"][0],params["img_shape"][1])).reshape(params["img_shape"])
        lbl_resized = cv2.resize(c_lbl, (params["img_shape"][0],params["img_shape"][1])).reshape(params["img_shape"][0]
                                                                                               ,params["img_shape"][0],1)
        # scaling
        label_im = np.divide(lbl_resized, 500., dtype=np.float32)
        train_im = np.divide(im_resized, 255., dtype=np.float32)
        # set all nans to zero
        label_im = np.nan_to_num(label_im)
        train_im = np.nan_to_num(train_im)

        # Train model on dataset
        prediction = model.predict(train_im.reshape(1,128,128,3))

        predicted_thickness_map = prediction[0,:,:,0]
        label = label_im[:,:,0]

        print("mae is:",np.abs(np.mean(predicted_thickness_map*500.-label*500.)))

        plt.imsave("./predictions/"+files_[i]+"_label.png",label*500.,cmap=plt.cm.jet)
        plt.imsave("./predictions/"+files_[i]+"_prediction.png",predicted_thickness_map*500.,cmap=plt.cm.jet)
