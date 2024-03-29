from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
import multiprocessing
from tensorflow.keras.utils.data_utils import get_file
import tensorflow as tf
import input as i
from utils import *
import model as m



def step_decay(epoch):
    # initial_lrate = 1.0 # no longer needed
    #drop = 0.5
    #epochs_drop = number_epoch_drop
    #lrate = init_lr * math.pow(drop,
    #math.floor((1+epoch)/epochs_drop))
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = learning_rate
    if epoch >= 60:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def load_models(model, params):

    if params.weights_init == 0:
        print("init random weights")

    if params.weights_init == 1:

        print("loading imagenet weights")

        WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                               'releases/download/v0.2/'
                               'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='a7b3fe01876f51b976af0dea6bc144eb')

        # get model weights
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        print("loaded imagenet model weights")

    if params.weights_init == 2:
        print("loading thickness_map")
        # get model weights
        model.load_weights(params.thickness_weights,
                           by_name=True,
                           skip_mismatch=True)

        print("load thickness map weights")

def main(logging ,params, step_factor):

    '''load model'''
    model = m.resnet_v2(params=params, input_shape=params.img_shape, n=params.depth,num_classes=5)

    '''load model weights'''
    if params.continue_training == 0:
        # get model weights
        load_models(model,params)
    else:
        model.load_weights(model_dir+"/weights.hdf5")
        print("loaded trained model under configuration")

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=params.learning_rate,momentum=0.99),
                  metrics=['accuracy'])

    #print model structure
    model.summary()

    #get standard configured data generators
    train_generator,valid_generator,test_generator = i.create_generators(params.data_path)

    #get data number of samples for training
    num_training_images, num_validation_images, num_test_images = i.get_data_statistics(params.data_path)

    '''callbacks'''
    lr_scheduler = LearningRateScheduler(step_decay)

    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath=logging.model_directory +"/weights.hdf5",
                                                    monitor='val_acc',
                                                    save_best_only=True,
                                                    verbose=1,
                                                    save_weights_only=False)

    tb = tensorflow.keras.callbacks.TensorBoard(log_dir=logging.tensorboard_directory,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True,
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)

    #saving model config file to model output dir
    logging.save_dict_to_json(logging.model_directory+"/config.json")

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(num_training_images/ (params.batch_size*1))*step_factor,
        epochs=params.num_epochs,
        validation_data=valid_generator,
        workers=multiprocessing.cpu_count() -3  ,
        use_multiprocessing=False,
        validation_steps=int(num_validation_images/ (1)),
        callbacks=[checkpoint, lr_scheduler, tb])
    
    pd.DataFrame(history.history).to_csv(logging.model_directory + "/loss_files.csv")
    


    print("###################### inititing predictions and evaluations ######################")
    pred = model.predict_generator(generator=test_generator,
                                         steps=int(num_test_images / (1)),
                                         verbose=1,
                                         use_multiprocessing=False,
                                         workers = 1)

    #get predictions and labels in list format
    preds = np.argmax(pred,axis=1).tolist()
    lbls = test_generator.labels.tolist()[:len(preds)]

    #instantiate the evaluation class
    evaluation = Evaluation(history=pd.DataFrame(history.history),
                            labels=lbls,
                            predictions=preds,
                            model_dir=logging.model_directory,
                            filenames=test_generator.filenames,
                            params=params)

    #get and save 5 example fundus images for each class in "./predictions and assemble to canvas"
    evaluation.plot_examples()
    evaluation.write_plot_evaluation()

#load utils classes
params = Params("params.json")

#set string attributes of params object
params.thickness_weights = "./thickness_model_weights/resnet_weights/weights.hdf5"
logging = Logging("./logs",params)


global num_filters_in
global learning_rate
global number_epoch_drop

number_epoch_drop = params.number_epoch_drop
learning_rate = params.learning_rate
num_filters = params.num_filters
# get model

# Training parameters that wont change
epochs = 1
data_augmentation = True
num_classes = 5

# Input image dimensions.
input_shape = params.img_shape

weights = ["random"]
data_cuts = ["512_100"]

for weight_config in weights:
    for dc in data_cuts:

        '''input pipeline'''
        params.data_path = "./data/" + dc
        params.model = "resnet_random_"

        #set model directory for saving all logs and models
        model_dir = logging.create_model_directory()

        if dc == "512_100":
            print("step factor is 1")
            step_factor = 1

        if dc == "512_50":
            print("step factor is 2")
            step_factor = 2

        if dc == "512_20":
            print("step factor is 5")
            step_factor = 5

        if dc == "512_10":
            print("step factor is 2")
            step_factor = 10

        if dc == "512_1":
            print("step factor is 2")
            step_factor = 100

        print("running config for:",weight_config)

        main(logging=logging,
             params = params,
             step_factor = step_factor)
