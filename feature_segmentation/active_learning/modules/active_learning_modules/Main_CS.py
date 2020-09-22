import os
import numpy as np
import pandas as pd
from paramsDRD import *
import DataGenerator as DG
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.optimizers import SGD
import ModelDRD as Model
import Uncertainty as UQ
import Evaluate as EV
import Load_data as loading
from kcenter_greedy_nalu import kCenterGreedy
from sklearn.model_selection import train_test_split
import utils

sys.dont_write_bytecode = True
np.set_printoptions(threshold=sys.maxsize)

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)
params["epochs"] = 100
params["save_path"] = "/storage/groups/ml01/projects/202003_AL_holmberg/transfer_learning/cycle_01/"
gen_params['image_path'] = params["data_dir"] + "train_images_mod2/"

'''Make new directories'''
utils.make_dirs(params["save_path"])
utils.make_dirs(params["save_path"] + "model_coreset/")

''' Create pandas dataframe for plots '''
df = pd.DataFrame(columns=['test_acc', 'train_acc', 'val_acc', 'test_loss', 'train_loss', 'val_loss', 'quadratic_kappa_score', 'weighted_accuracy', 'n_examples'])
df_ind = pd.DataFrame()
df_pred = pd.DataFrame()

''' Load data '''
df_data = pd.read_csv(gen_params["label_path"], dtype=str)
train_val_df, test_df = train_test_split(df_data, test_size=0.2, stratify=df_data["diagnosis"], random_state=42)  # test = 20%
train_df, val_df = train_test_split(train_val_df, test_size=0.15, stratify=train_val_df["diagnosis"], random_state=42)  # val = 10%

''' Calculate and load parameters '''
budget = int(params["cycle_length"] * len(train_df)) + 1
iterations = int((len(train_df) - params["cycle_length"] * len(train_df)) / budget) - 2
opt = SGD(lr=params["learning_rate"], momentum=0.9)
lr_scheduler = LearningRateScheduler(utils.step_decay)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

''' Divide into labeled and unlabeled '''
[labelled_df, unlabelled_df, ind_to_label] = loading.divide_data(train_df, budget)
df_ind["iter_{}".format(0)] = ind_to_label

''' Add filename extension .png '''
val_df["id_code"] = val_df["id_code"].apply(utils.append_ext)
test_df["id_code"] = test_df["id_code"].apply(utils.append_ext)

''' Load and test generator '''
test_generator = DG.create_test_generator(test_df)
val_generator = DG.create_val_generator(val_df)

for iter in range(iterations):

    ratio = round(len(labelled_df)/len(train_df), 2)
    print('-------------- Ratio: {} --------------'.format(ratio))

    ''' Break for loop at a specific iteration'''
    if ratio == 1.1:
        break

    ''' Generate generators '''
    labelled_generator = DG.create_generator(labelled_df)
    if iter < iterations - 2:
        unlabelled_generator = DG.create_test_generator(unlabelled_df, params["batch_size_eval"])

    ''' Compile model '''
    model = Model.ResNet50(is_training=True, include_top=True,  weights=None, input_tensor=None, input_shape=params["input_shape"], pooling="avg", classes=5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    ''' Initialize callbacks '''
    save_model_path = os.path.join(params["save_path"] + "model_coreset/", "model_iteration_{}.hdf5".format(iter))
    cp = ModelCheckpoint(filepath=save_model_path, monitor='val_acc', save_best_only=True, verbose=0, mode='max')
    csv_logger = CSVLogger(filename=params["save_path"] + 'history_coreset_{}.csv'.format(iter), append=True, separator=",")

    if params["continuing_training"]:
        '''Load models trained weights'''
        model.load_weights(save_model_path)

    if iter > 2:
        epochs = params["epochs"]
    else:
        epochs = params["epochs"] + 15

    training_hist = model.fit(labelled_generator,
                              steps_per_epoch=len(labelled_df) // params["batch_size"],
                              validation_data=val_generator,
                              validation_steps=len(val_df) // params["batch_size"],
                              epochs=epochs,
                              callbacks=[cp, lr_scheduler, lr_reducer, csv_logger],
                              workers=8,
                              verbose=2)

    [model_eval, test_acc, test_loss, weighted_accuracy, kappa_score, pred] = EV.evaluate(opt, test_generator, test_df,
                                                                                          "model_coreset/model_iteration_{}.hdf5".format(iter),
                                                                                          "cm_coreset_{}".format(iter))

    ''' Output evaluation accuracy for plot '''
    df = df.append({'test_acc': test_acc, 'train_acc': training_hist.history['acc'], 'val_acc': training_hist.history['val_acc'],
                    'test_loss': test_loss, 'train_loss': training_hist.history['loss'], 'val_loss': training_hist.history['val_loss'],
                    'quadratic_kappa_score': kappa_score, 'weighted_accuracy': weighted_accuracy, 'n_examples': ratio}, ignore_index=True)
    df_pred["iter_{}".format(iter)] = pred

    if iter < iterations - 1:
        ''' Get post-softmax layer '''
        psoftmax = UQ.get_postsoftmax(model_eval, unlabelled_generator, len(unlabelled_df))

        ''' Apply core-set algorithm to get 'budget' examples to label '''
        if len(unlabelled_df) > budget:
            kcenters = kCenterGreedy(psoftmax)
            [ind_to_label, min_dist] = kcenters.select_batch_(already_selected=kcenters.already_selected, N=budget)
        else:
            ind_to_label = np.arange(len(unlabelled_df))
        df_ind["iter_{}".format(iter+1)] = ind_to_label

        ''' Reorganize training data into labelled and unlabelled '''
        labelled_df = labelled_df.append(unlabelled_df.iloc[ind_to_label], ignore_index=True)
        unlabelled_df.drop(unlabelled_df.index[ind_to_label], inplace=True)
        psoftmax = np.delete(psoftmax, ind_to_label, axis=0)

''' Output results to text files '''
f = open(os.path.join(params["save_path"], "params_coreset.csv"), "a+")
f.write(str(params))
f.close()
df.to_csv(params["save_path"] + 'acc_plot_coreset.csv')
df_ind.to_csv(params["save_path"] + 'ind_to_label_coreset.csv')
df_pred.to_csv(params["save_path"] + 'predictions_coreset.csv')






