import os
import model_test as mt
from train_eval_ops import *
from keras.optimizers import *

from keras.layers import Input
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import callbacks
import loading_numpy_functions as lnf
from params import params
sess = tf.InteractiveSession()

'''create save dir if does not exist'''
try:
    os.stat(params["save_path"])
except:
    os.makedirs(params["save_path"])

'''load data files'''
path = "/storage/groups/ml01/datasets/projects/20181610_eyeclinic_niklas.koehler/oct_paths.csv"
oct_paths = pd.read_csv(path)[[0]]
num_training_examples = oct_paths.shape[0]
'''get model'''
input_img = Input(params["img_shape"], name='img')
model = mt.get_bunet(input_img, n_filters=16, dropout=0.0, batchnorm=True,training=False)

adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss=generalized_dice_loss)

model.summary()

'''train and save model'''
save_model_path = os.path.join(params["save_path"],"weights.hdf5")
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor="generalized_dice_loss",
                                        save_best_only=True, verbose=1,save_weights_only=True)

learning_rate_reduction = callbacks.ReduceLROnPlateau(factor=0.1, patience=10,min_lr=0.0001)

model.load_weights(save_model_path)

from sklearn.metrics import classification_report

target_names = ["0", "1", "2", "3", "4", "5","6", "7", "8", "9", "10"]


image_iter = 0
for i in oct_paths.valuesi[0:10]:

    if image_iter % int(num_training_examples/params["batch_size"]) == 0:
        image_iter = 0

    #select batch size of image paths
    im = cv2.imread(os.path.join(i),0)
    im = cv2.resize(im,(256,256),interpolation=cv2.INTER_NEAREST)
    im = np.divide(im, 255., dtype=np.float32)
    
    image_iter += 1
    prediction = model.predict(im_batchi.reshape(1,256,256,1))
    pred_ = np.argmax(prediction.reshape([-1, 11]), axis=-1)
    pred_map = pred_.reshape(256, 256)
    print(np.unique(pred_map,return_counts=True))
