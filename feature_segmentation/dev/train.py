from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
import random
from utils import Params, TrainOps, Evaluation, Logging
from model import unet
import os
from python_generator import DataGenerator
import glob
import numpy as np
tf.compat.v1.disable_eager_execution()
import cv2
from DeepLab3plus import Deeplabv3
import segmentation_models as sm
import networks.SEunet as su
from networks.wBiFPN import wBi_unet

# load utils classes
params = Params("params.json")
logging = Logging("./logs", params)
trainops = TrainOps(params)

# create logging directory
logging.create_model_directory()

ids = os.listdir(params.data_path + "/images")

#test_ = os.listdir("./data/old_versions/test_iteration_processing/annotations/ben")

#test_ids = []
## remove test ids
#for id in ids:
#    if id.replace(".json.png", "") in test_:
#        ids.remove(id)
#        test_ids.append(id)


random.shuffle(ids)

train_ids = ids[0:int(len(ids) * 0.8)][0:1]*50
validation_ids = train_ids[0:1]#ids[int(len(ids) * 0.8) + 1:-1]
test_ids = train_ids#ids[int(len(ids) * 0.8) + 1:-1]

partition = {'train': train_ids,
             'validation': validation_ids,
             'test': test_ids}

print("number of train, validation and test image are: ", len(train_ids), len(validation_ids), len(test_ids))

params.is_training = True
params.model_directory = logging.model_directory

# log train files
np.savetxt(os.path.join(params.model_directory, "train_images.csv"), partition['train'], fmt = "%s")
np.savetxt(os.path.join(params.model_directory, "validation_images.csv"),  partition['validation'], fmt = "%s")
np.savetxt(os.path.join(params.model_directory, "test_images.csv"),  partition['test'], fmt = "%s")

# saving model config file to model output dir
logging.save_dict_to_json(logging.model_directory + "/config.json")

# Generators
train_generator = DataGenerator(partition['train'], params = params, is_training = True)


x,y = train_generator.__getitem__(0)
# Generators
test_generator = DataGenerator(partition['validation'], params = params, is_training = False)

# plot examples
for k in range(1):
    trainops.plot_examples(train_generator.example_record(), "train_{}".format(k))

trainops.plot_examples(test_generator.example_record(), "test_1")
trainops.plot_examples(test_generator.example_record(), "test_2")

'''get model'''
model = unet(params)

'''Compile model'''
model.compile(optimizer = Adam(lr = params.learning_rate),
              loss = "sparse_categorical_crossentropy",
              metrics = ['accuracy', sm.metrics.iou_score])

model.summary()

'''train and save model'''
save_model_path = os.path.join(logging.model_directory, "weights.hdf5")

# load model
# model.load_weights("./logs/13/weights.hdf5")

# Train model on dataset
history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = int(len(train_ids) / (params.batch_size * 1)),
                              epochs = params.num_epochs,
                              validation_data = test_generator,
                              validation_steps = int(len(test_ids) / 1),
                              callbacks = trainops.callbacks_())
