from train_eval_ops import *
import tensorflow as tf
from keras.optimizers import adam
import model as mt
import os
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from params import *
from python_generator import DataGenerator
from train_eval_ops import *
import pandas as pd
import sys
import resnet as re

sys.dont_write_bytecode = True

train_file_names = "./full_export_file_names/y_train_filenames_filtered.csv"
validation_file_names = "./full_export_file_names/y_validation_filenames_filtered.csv"

train_ids = pd.read_csv(train_file_names)["0"]
validation_ids = pd.read_csv(validation_file_names)["0"]


num_train_examples = train_ids.shape[0]
num_val_examples = validation_ids.shape[0]

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))

partition = {'train': train_ids.values.tolist(), 'validation': validation_ids.values.tolist()}

# Generators
training_generator = DataGenerator(partition['train'], is_training=True, **gen_params)
validation_generator = DataGenerator(partition['validation'], is_training=False,**gen_params)

res_output,img_input = re.resnet_v2(params["img_shape"], 1)
outputs = mt.decoder(res_output, n_filters=16, dropout=0.05, batchnorm=True)
model = Model(inputs=img_input, outputs=[outputs])
'''Compile model'''
adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss=custom_mae, metrics=[custom_mae,percentual_deviance])
model.summary()

'''train and save model'''
save_model_path = os.path.join(params["save_path"], "weights.hdf5")
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_percentual_deviance',
                                        save_best_only=True, verbose=1, save_weights_only=True)

es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=50)

learning_rate_reduction = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1)

if params["continuing_training"] == True:
    '''Load models trained weights'''
    model.load_weights(save_model_path)

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = int(num_train_examples/(params["batch_size"])),
                    validation_steps=int(num_train_examples/(params["batch_size"])),
                    epochs=params["epochs"],
                    verbose = 1,
                    workers=4,
                    callbacks=[cp, learning_rate_reduction])

pd.DataFrame(history.history).to_csv("loss_curves.csv")
