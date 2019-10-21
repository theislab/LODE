from train_eval_ops import *
import model as m
import os
from params import *
from keras.models import *
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from keras.optimizers import adam
from train_eval_ops import *
import cv2
from sklearn.metrics import accuracy_score
import resnet as re

id_path = "./file_splits"

test = pd.read_csv(os.path.join(id_path,"test.csv"))

num_test_examples = test.shape[0]

print("Number of test examples: {}".format(num_test_examples))

partition = {'test': np.unique(test["image"]).tolist()}
# get model
# get model
res_output, img_input = re.ResNet50(params["img_shape"], 5)
model = Model(inputs=img_input, outputs=[res_output])
model.summary()
'''Compile model'''
Adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=Adam, loss="categorical_crossentropy",
              metrics=["accuracy", fmeasure])

#load model weights
save_model_path = os.path.join(params["save_path"], "weights.hdf5")
'''Load models trained weights'''
model.load_weights(save_model_path)

lbls = []
preds = []

for i in range(0,len(partition['test'])):
    #get id record
    id_name = partition["test"][i]
    #get image
    scale=300
    im = cv2.imread(os.path.join(gen_params["fundus_path"],id_name+".jpeg"))

    x = im[int(im.shape[0] / 2), :, :].sum(1)  # sum over axis=1
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    mod_img = cv2.resize(im , (0, 0), fx=s, fy=s)

    # substract local mean color
    blurred = cv2.GaussianBlur(mod_img, (0, 0), scale / 30)
    mod_img = cv2.addWeighted(mod_img, 4, blurred, -4, 128)

    # remove outer 10%
    b = np.zeros(mod_img.shape, mod_img.dtype)
    cv2.circle(b, (int(mod_img.shape[1] / 2), int(mod_img.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    mod_img = (mod_img * b) + 128 * (1 - b)

    mod_img = mod_img / 255.

    im = im / 255.
    im = cv2.resize(im, (params["img_shape"][0],params["img_shape"][1]))

    #get label
    label = test[test["image"] == id_name]['level'].iloc[0]
    #predict
    predictions = model.predict(im.reshape(1,params["img_shape"][0],params["img_shape"][1],3))
    #getting prediction
    prediction = np.argmax(predictions)
    #log
    lbls.append(label)
    preds.append(prediction)

    if i % 100 == 0:
        print "number of processed images are",i

print(params["save_path"])
print("Accuracy score:",accuracy_score(lbls, preds))
print("Recall score:",recall_score(lbls, preds,average='macro'))
print("Precision score:",precision_score(lbls, preds,average='macro'))
print("confusion matrix:",confusion_matrix(lbls, preds))






