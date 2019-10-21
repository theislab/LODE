from train_eval_ops import *
import model_test as mt
import os
from params import *
from keras.models import *
import data_generator as dg
import pandas as pd
import sys
from keras.optimizers import adam
from train_eval_ops import *
import deeplab3pluss as dl
import cv2
import regular_unet as ru
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers


from keras.models import Model
import jupyter_helper_functions as jh

sys.dont_write_bytecode = True

test_pd_path = "/home/olle/PycharmProjects/thickness_map_prediction/\
project_evaluation/doctor_panel_test/gold_standard_answers/joint_gold_standard.csv"

test_pd  = pd.read_csv(test_pd_path)

test_ids = test_pd.record_name.str.replace(".png","").values

'''
test_ids = [i.replace(".png","").replace(".npy","") for i in os.listdir(params["data_dir"])]
test_ids = list(set(test_ids))
test_ids_pd = pd.read_csv("./stratified_and_patient_split_full_export/x_test_filenames_filtered.csv")

test_ids = [i.replace(".png","") for i in test_ids_pd["0"].values]

records_to_eval_path = "./records_to_uncertainty_evaluate.csv"
eval_ids = pd.read_csv(records_to_eval_path)["0"]
'''

#################################### --- LOAD MOEL --- #################################################################

def instantiate_bunet(params,adam):

    input_img = Input(params["img_shape"], name='img')

    model = mt.get_bunet(input_img, n_filters=16, dropout=0.05, batchnorm=True, training=False)

    '''Compile model'''
    adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam, loss="mse", metrics=["mse"])#,percentual_deviance])
    model.summary()

    '''train and save model'''
    save_model_path = os.path.join(evaluation_params["model_path"], "weights.hdf5")
    '''Load models trained weights'''
    model.load_weights(save_model_path)

    return model

def instantiate_Deeplabv3(params, adam):

    input_img = Input(params["img_shape"], name='img')

    model = dl.Deeplabv3(input_tensor=input_img, input_shape=params["img_shape"], training=False, OS=16)

    '''Compile model'''
    adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam, loss=mse, metrics=[mse,percentual_deviance])
    model.summary()

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"], "weights.hdf5")

    '''Load models trained weights'''
    model.load_weights(save_model_path)

    return model

def instantiate_regular_unet(params):
    inputs, outputs = ru.model_fn(params["img_shape"])
    model = models.Model(inputs=inputs, outputs=[outputs])

    '''Compile model'''
    adam = optimizers.Adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam, loss="mse", metrics=["mse"])
    model.summary()

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"], "weights.hdf5")

    '''Load models trained weights'''
    model = models.load_model(save_model_path)

    return model

model = instantiate_bunet(params,adam)

def get_input(path):
    img = cv2.resize(cv2.imread(path,0).astype(np.float32),(params["img_shape"][1],params["img_shape"][2]))
    return (img)

def get_output(path):
    labels = cv2.resize(np.load(path).astype(np.float32),(params["img_shape"][1],params["img_shape"][2]))
    return (labels)

def pre_process(train_im, label_im):
    # remove non labeled part of image
    train_im[label_im == 0] = 0
    label_im[label_im == 0] = 0

    # remove non labeled part of label
    train_im[train_im== 0] = 0
    label_im[train_im== 0] = 0

    # scaling
    label_im = np.divide(label_im, 500., dtype=np.float32)
    train_im = np.divide(train_im, 255., dtype=np.float32)
    # set all nans to zero
    label_im = np.nan_to_num(label_im)
    train_im = np.nan_to_num(train_im)
    return(train_im.reshape(params["img_shape"][1],params["img_shape"][2],1),
           label_im.reshape(params["img_shape"][1],params["img_shape"][2],1))

def evaluation_load(im_path, lbl_path):
    input = get_input(im_path)
    output = get_output(lbl_path)
    input_, output_ = pre_process(input, output)
    return(input_.reshape(1,params["img_shape"][1],params["img_shape"][2],1),
           output_.reshape(1,params["img_shape"][1],params["img_shape"][2],1))

def pixel_wise_aleatoric_loss(variance, y_true, y_pred,i):
    '''
    :param variance:
    :param y_true:
    :param y_pred:
    :return:
    '''
    var_flat = variance.flatten()
    pixel_perc = np.divide(np.abs(y_true[0,:,:,0] - y_pred[0,:,:,0]),y_true[0,:,:,0])
    pixel_perc[pixel_perc == np.inf] = 0
    pixel_flat = pixel_perc.flatten()

    return(pd.DataFrame([[i]*pixel_flat.shape[0],var_flat.tolist(),pixel_flat.tolist()]))

def crop_image(img,cond, tol=0):
    # img is image data
    # tol  is tolerance
    mask = cond>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def get_image_and_label(gen_params,i):
    im_path = os.path.join(gen_params["fundus_path"], i + ".png")
    lbl_path = os.path.join(gen_params["thickness_path"], i + ".npy")
    im , lbl = evaluation_load(im_path, lbl_path)
    return(im,lbl)

def seperate_pred_var(pred):
    return pred[:, :, :, 0], pred[:, :, :, 1]

def rescale_pred_lbl(pred, lbl):
    return(pred*500.,lbl*500.)

def get_avg_thickness(y_true):
    return np.mean(y_true[y_true != 0])

def append_to_list(perc_dev, mae, loss, result_log,avg_thickness,i):
    result_log[0].append(perc_dev)
    result_log[1].append(mae)
    result_log[2].append(loss)
    result_log[3].append(avg_thickness)
    result_log[4].append(i)
    return(result_log)

def log_general_performance_metrics(record_log_list):
    # convert result log to pandas data frame and name columns
    result_log_pd = pd.DataFrame(record_log_list).T
    result_log_pd = result_log_pd.rename(columns={0: "percentual deviance", 1: "mae", 2: "uncertainty loss",
                                                  3: "aleatoric uncertainty", 4: "average_ground_truth_thickness",
                                                  5: "patient ID"})
    # save table
    save_log_path = "/home/olle/PycharmProjects/thickness_map_prediction/analysis/general_performance_metrics_prevmodel.csv"
    result_log_pd.to_csv(save_log_path, index=False)

def log_aleatoric_pixel_evaluation(aleatoric_pd):
    aleatoric_pd.to_csv("./analysis/aleatoric_log.csv")

def save_aleatoric_map(variance, label,evaluation_params, i):
    #remove extra axises
    var = variance[0,:,:]
    lbl = label[0,:,:,0]
    # crop image
    var = crop_image(var, lbl)
    np.save(evaluation_params["save_aletoric_record_path"],i +".npy",var)

def generate_aleatoric_analysis(params,test_ids):
    '''
    :param params:
    :param test_ids:
    :return:
    '''
    #assemble the first result log
    aleatoric_pd = pd.DataFrame()
    for k,i in enumerate(test_ids):

        #load image
        im , lbl = get_image_and_label(gen_params,i)

        #predict
        pred = model.predict(im.reshape(1, params["img_shape"][0],
                                        params["img_shape"][1],
                                        params["img_shape"][2]),
                                        batch_size=1)

        #seperate prediction and variance
        prediction, variance = seperate_pred_var(pred)

        #assemble and log pixel wise aseatoric uncertainty together with percentual deviance / loss
        pd_temp = pixel_wise_aleatoric_loss(variance, lbl, pred,i)
        aleatoric_pd = aleatoric_pd.append(pd_temp.T)

        # if save aleatoric map
        if evaluation_params["save_aleatoric"]:
            save_aleatoric_map(variance, lbl, evaluation_params, i)

    #write log to dataframe file
    log_aleatoric_pixel_evaluation(aleatoric_pd)

def generate_main_metrics(params,test_ids,model, save_pred):
    '''
    :param params:
    :param test_ids:
    :return:
    '''
    from skimage.transform import resize
    #assemble the first result log
    aleatoric_pd = pd.DataFrame()
    result_log = [[],[],[],[],[]]
    for k,i in enumerate(test_ids):
        #load image
        im , lbl = get_image_and_label(gen_params,i)
        #predict
        pred = model.predict(im.reshape(1,
                                        params["img_shape"][1],
                                        params["img_shape"][2],
                                        1),
                                        batch_size=1)

        if save_pred:
            np.save(os.path.join(params["save_predictions_dir"],i + ".npy"), pred[0,:,:,0])

        #general performance metricy
        # -- percentual deviance
        perc_dev = percentual_deviance_np(lbl, pred)
        # -- mae
        mae = custom_mae_np(lbl, pred)
        # -- loss
        loss = uncertainty_loss_mae_np(lbl, pred)
        # -- avg thickness
        avg_thickness = get_avg_thickness(lbl)

        #append metric values to list for logging
        result_log = append_to_list(perc_dev, mae, loss, result_log, avg_thickness, i)
        #print("Record not working is: {}".format(i))
    log_general_performance_metrics(result_log)


def sample_espistemic_loss(params,IDs):
    '''
    :param IDs: record ids to be evaluated
    :param T: number of times to sample the network weights
    :return: a nested list with each row being a record and each sublist
    a sample
    Inspired by paper: https://arxiv.org/pdf/1703.04977.pdf
    '''
    from skimage.transform import resize
    # init nested list
    epi_variances = []
    epi_record_id = []
    # retrieve each sample

    # load model with dropout in training mode
    input_img = Input(params["img_shape"], name='img')
    epistemic_model = mt.get_bunet(input_img, n_filters=16, dropout=0.5, batchnorm=True, training=True)

    '''train and save model'''
    save_model_path = os.path.join(evaluation_params["model_path"], "weights.hdf5")

    '''Load models trained weights'''
    epistemic_model.load_weights(save_model_path)

    print("loaded model with dropout in training mode")
    for i in IDs:
        im_path = os.path.join(gen_params["fundus_path"], i + ".png")
        lbl_path = os.path.join(gen_params["thickness_path"], i + ".npy")
        im, lbl = evaluation_load(im_path, lbl_path)

        pred = epistemic_model.predict(im.reshape(1,params["img_shape"][0],
                                        params["img_shape"][1], 1), batch_size=1)

        #get only uncertainties for fovea region
        prediction = pred[0, :, :, 0].flatten().tolist()

        epi_variances.extend(prediction)
        epi_record_id.extend([i]*len(prediction))

    return (epi_variances, epi_record_id)



def epistemic_analysis(params,test_ids):
    # running the epistemic loss functions
    T = 10
    # save table
    save_epistemic_pd_path = "./analysis/"
    epistemic_pd = pd.DataFrame(columns={"0","1","2","3","4","5","6","7","8","9"})
    for j in range(T):
        #save_name = "epistemic_uncertainty_sample_{}.csv".format(str(j))
        epi_variances, epi_record_id = sample_espistemic_loss(params, test_ids)
        epistemic_pd[[str(j)]] = pd.DataFrame(epi_variances)
        epistemic_pd["record_id"] = pd.DataFrame(epi_record_id)

        print("Completed {}th sampling".format(j))

    epistemic_pd.to_csv(save_epistemic_pd_path + "epistemic_uncertainty.csv")


#general metric
generate_main_metrics(params, test_ids, model, save_pred=True)
#aleatoric analysis
#generate_aleatoric_analysis(params,test_ids)
#epistemic analyis
#epistemic_analysis(params,test_ids)







