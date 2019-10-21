from __future__ import print_function
from keras.optimizers import SGD
import input as i
import glob
from utils import *
import model as m
import pandas as pd

model_dirs = ["/home/olle/PycharmProjects/thickness_map_prediction/ssl_kaggle/server_logs/server_logs_v3/50_partition/ssl_50_iter1"]

for model_dir in model_dirs:
    history = pd.read_csv(os.path.join(model_dir, "loss_files.csv"))

    # load utils classes
    params = Params("params.json")

    '''load model'''
    model = m.resnet_v2(params=params, input_shape=params.img_shape, n=params.depth, num_classes=5)

    model.load_weights(model_dir + "/weights.hdf5")
    print("loaded trained model under configuration")

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=params.learning_rate, momentum=0.99),
                  metrics=['accuracy'])

    '''input pipeline'''
    params.data_path = "/media/olle/Seagate/kaggle/keras_format/pre_proc/512_10"

    # get standard configured data generators
    test_generator = i.create_test_generator(params.data_path)

    num_test_images = i.get_test_statistics(params.data_path)


    print("###################### inititing predictions and evaluations ######################")
    pred = model.predict_generator(generator=test_generator,

                                         steps=int(num_test_images / (1)),
                                         verbose=1,
                                         use_multiprocessing=False,
                                         workers = 5)

    #get predictions and labels in list format
    preds = np.argmax(pred,axis=1).tolist()
    lbls = test_generator.labels.tolist()

    pd.DataFrame(preds).to_csv(model_dir + "/predictions.csv")
    pd.DataFrame(lbls).to_csv(model_dir + "/labels.csv")

    #instantiate the evaluation class
    evaluation = Evaluation(history=history,
                            labels=lbls[:len(preds)],
                            predictions=preds,
                            softmax_output=pred,
                            model_dir=model_dir,
                            filenames=test_generator.filenames,
                            params=params)


    evaluation.write_plot_evaluation()
    evaluation.plot_examples()
