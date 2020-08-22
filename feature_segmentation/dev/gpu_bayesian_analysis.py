import pandas as pd
from train_eval_ops import *
from keras.optimizers import adam
from loading_numpy_functions import *

def predictive_uncertainty(y_pred_mc):
    '''
    :param y_pred_mc: number of pixels x number of classes x  number of MC samples
    :return:
    '''
    #FOR EACH CLASS, GET mc AVERGARE OF PROB. THEN MULTIPLY REG WITH LOG OF AVERAGE AND SUM OVER CLASSES

    return -np.mean(np.sum(y_pred_mc*np.log(y_pred_mc),axis=(1)),axis=(1)).reshape(256,256)

#mutual information - epistemic uncertainty
def predictive_entropy(y_pred):
    '''
    :param y_pred: y_pred_mc:number of pixels x number of classes x  number of MC samples
    :return:
    '''
    mc_mean = np.mean(y_pred,axis=-1)
    mc_log_mean = np.log(mc_mean)
    return -1*np.sum(mc_mean*mc_log_mean,axis=1).reshape(256,256)

# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)

def generate_predictive_entropy_uncertainty_map(model, image, num_mc_samples):
    '''
    :param model: keras model object
    :param image: ivoct image
    :param label: label
    :param num_mc_samples: integer
    :return: a label sized image containg per pixel mutual information
    '''
    count = 0
    M2 = np.empty((params["img_shape"][0]*params["img_shape"][1],params["number_of_classes"]))
    mean = np.empty((params["img_shape"][0]*params["img_shape"][1],params["number_of_classes"]))
    existingAggregate = count, mean, M2
    #generate samples
    for i in range(num_mc_samples):
        pred = model.predict(image)
        existingAggregate = update(existingAggregate, pred.reshape([-1,params["number_of_classes"]]))

    bayes_mean, bayes_var, bayes_sampleVar = finalize(existingAggregate)
    return np.argmax(bayes_mean, axis=-1).reshape(256,256)\
        ,np.mean(bayes_var, axis=-1).reshape(256,256)\
        ,np.mean(bayes_sampleVar, axis=-1).reshape(256,256)

ids_train = pd.read_csv(os.path.join(params["data_dir"], "train_records.csv"))
ids_val = pd.read_csv(os.path.join(params["data_dir"], "validation_records.csv"))
ids_test = pd.read_csv(os.path.join(params["data_dir"], "test_records.csv"))

#make to list
ids_train = ids_train.values.flatten().tolist()
ids_val = ids_val.values.flatten().tolist()
ids_test = ids_test.values.flatten().tolist()


num_training_examples = len(ids_train)
num_val_examples = len(ids_test)

print("Number of training examples: {}".format(num_training_examples))
print("Number of validation examples: {}".format(num_val_examples))

partition = {'train': ids_train, 'validation': ids_val, 'test': ids_test}

#init model
model_fixed = instantiate_bunet(params, adam, training=False)
model_mc = instantiate_bunet(params, adam, training=True)

for k,i in enumerate(partition["test"]):

    #load image
    im , lbl = get_image_and_label(gen_params,i)
    #predict
    pred = model_fixed.predict(im)
    pred_ = pred.reshape(-1, params["number_of_classes"])
    pred_mask = np.argmax(pred_, axis=1).reshape(256, 256)

    eval = model_fixed.evaluate(im,lbl)

    bayes_mean, bayes_var, bayes_sampleVar = generate_predictive_entropy_uncertainty_map(model_mc,im, num_mc_samples=1000)

    np.save("./predictions/{}".format(i) + "_bayes_prediction.npy", bayes_mean)
    np.save("./predictions/{}".format(i) + "_bayes_variance.npy", bayes_var)
    np.save("./predictions/{}".format(i) + "_reg_pred.npy", pred_mask)
    np.save("./predictions/{}".format(i) + "_ground_truth.npy", lbl[0,:,:,0])
