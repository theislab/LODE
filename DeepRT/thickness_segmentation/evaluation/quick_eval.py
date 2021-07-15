from train_eval_ops import *
from input import *
import os
import numpy as np
import tensorflow as tf
from model import *
from tensorflow.python.tensorflow.keras import models

def quick_eval(params,save_model_path):
    img_dir = os.path.join(params["data_dir"], "test_images")
    label_dir = os.path.join(params["data_dir"], "test_labels")

    ids_train = [i.replace(".png","") for i in os.listdir(img_dir)]

    x_train_filenames = []
    y_train_filenames = []
    for img_id in ids_train:
      x_train_filenames.append(os.path.join(img_dir, "{}.png".format(img_id)))
      y_train_filenames.append(os.path.join(label_dir, "{}.png".format(img_id)))

    y_val_filenames = y_train_filenames
    x_val_filenames = x_train_filenames

    num_train_examples = len(x_train_filenames)
    num_val_examples = len(x_val_filenames)

    print("Number of training examples: {}".format(num_train_examples))
    print("Number of validation examples: {}".format(num_val_examples))

    # Alternatively, load the weights directly: model.load_weights(save_model_path)
    from tensorflow.keras.models import load_model
    inputs, outputs = model_fn(params["img_shape"])
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.load_weights(save_model_path)
    train_ds,val_ds,temp_ds = batch_data_sets(x_train_filenames,y_train_filenames,x_val_filenames,y_val_filenames)

    # Let's visualize some of the outputs
    data_aug_iter = val_ds.make_one_shot_iterator()
    next_element = data_aug_iter.get_next()

    # Running next element in our graph will produce a batch of images
    test_losses = []
    for i in range(12):
        batch_of_imgs, label = tf.tensorflow.keras.backend.get_session().run(next_element)
        img = batch_of_imgs[0]
        loss = model.evaluate(x=batch_of_imgs, y=label, batch_size=1, verbose=1, sample_weight=None, steps=None)
        test_losses.append(loss)

    print("average dice score over all samples are:{}".format(np.mean(test_losses)))

params = {}
params["img_shape"] = (512, 512, 3)
params["batch_size"] = 3
params["epochs"] = 5
params["data_dir"] = "/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/data/clinic_data"
params["save_path"] = "/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/output/dice_loss_0.1_256"

model_modes = os.listdir(params["save_path"])

for modes in model_modes:
    save_model_path = os.path.join(params["save_path"],"weights.hdf5")
    quick_eval(params,save_model_path)