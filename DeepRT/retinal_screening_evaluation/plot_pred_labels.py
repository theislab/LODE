import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from jupyter_helper_functions import *

label_path = "/home/olle/PycharmProjects/thickness_map_prediction/\
project_evaluation/test_labels/"
prediction_path = "/home/olle/PycharmProjects/thickness_map_prediction/\
project_evaluation/test_predictions"
image_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/test_images"

def get_text_coord():
    C0, S2, S1, N1, N2, I1, I2, T1, T2 = get_low_res_grid()
    C0_x_mc = np.median(np.where(C0 == True)[0])
    C0_y_mc = np.median(np.where(C0 == True)[1])

    S1_x_mc = np.median(np.where(S1 == True)[1])
    S1_y_mc = np.median(np.where(S1 == True)[0])

    S2_x_mc = np.median(np.where(S2 == True)[1])
    S2_y_mc = np.median(np.where(S2 == True)[0])

    N1_x_mc = np.median(np.where(N1 == True)[1])
    N1_y_mc = np.median(np.where(N1 == True)[0])

    N2_x_mc = np.median(np.where(N2 == True)[1])
    N2_y_mc = np.median(np.where(N2 == True)[0])

    I1_x_mc = np.median(np.where(I1 == True)[1])
    I1_y_mc = np.median(np.where(I1 == True)[0])

    I2_x_mc = np.median(np.where(I2 == True)[1])
    I2_y_mc = np.median(np.where(I2 == True)[0])

    T1_x_mc = np.median(np.where(T1 == True)[1])
    T1_y_mc = np.median(np.where(T1 == True)[0])

    T2_x_mc = np.median(np.where(T2 == True)[1])
    T2_y_mc = np.median(np.where(T2 == True)[0])
    coord_list = [C0_x_mc, C0_y_mc, S2_x_mc, S2_y_mc, S1_x_mc, S1_y_mc, \
                  N1_x_mc, N1_y_mc, N2_x_mc, N2_y_mc, I1_x_mc, I1_y_mc, I2_x_mc, I2_y_mc, \
                  T1_x_mc, T1_y_mc, T2_x_mc, T2_y_mc]
    return (coord_list)


def load_images(record_name):
    label = np.load(os.path.join(label_path, record_name))
    prediction = np.load(os.path.join(prediction_path, record_name))
    image = cv2.imread(os.path.join(image_path, record_name))
	
    label_mu = pixel_to_mu_meter(label)
    prediction_mu = pixel_to_mu_meter(prediction)
    # resize prediciton
    prediction_mu = resize_prediction(prediction_mu)

    # remove three channel to stop normalization
    label_lr = get_low_res_depth_grid(label_mu)[:, :, 0]
    prediction_lr = get_low_res_depth_grid(prediction_mu)[:, :, 0]
    return (label_mu, prediction_mu, label_lr, prediction_lr,image)


def plot_pred_label_pair(record_name):
    label_mu, prediction_mu, label_lr, prediction_lr,image = load_images(record_name)
    coord_list = get_text_coord()
    plt.figure(figsize=(20, 20))
    plt.subplot(3, 2, 1)
    plt.imshow(label_mu)
    plt.title("label:{}".format(record_name))
    plt.colorbar()
    plt.subplot(3, 2, 2)
    plt.imshow(prediction_mu)
    plt.title("prediction")
    plt.colorbar()

    plt.subplot(3, 2, 3)
    plt.autoscale(True)
    plt.imshow(label_lr)
    plt.title("label low res:{}".format(record_name))
    for i in range(0, int(len(coord_list) / 2)):
        print(coord_list[i * 2], coord_list[(i+1) * 2 - 1])
        plt.text(coord_list[i * 2], coord_list[i * 2 - 1], "test", ha='center', va='center')
    plt.colorbar()
    plt.subplot(3, 2, 4)
    plt.autoscale(True)
    plt.imshow(prediction_lr)
    plt.title("prediction low res")
    plt.colorbar()

    plt.subplot(3, 2, 5)
    plt.autoscale(True)
    plt.imshow(image)
    plt.title("fundus")
    plt.show()

record_name = "763_2010-10-27_R_08-40-09.npy"
plot_pred_label_pair(record_name)
