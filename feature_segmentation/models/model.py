from keras.engine.saving import load_model
from keras.optimizers import adam
from feature_segmentation.models.networks import standard_unet, deep_unet, SEdeep_unet, deeper_unet, volumeNet
import segmentation_models as sm


def get_model(params):

    global model
    if params.model == 'standard_unet':
        model = standard_unet.unet(params)

    if params.model == 'deep_unet':
        model = deep_unet.unet(params)

    if params.model == 'deeper_unet':
        model = deeper_unet.unet(params)

    if params.model == 'SEdeep_unet':
        model = SEdeep_unet.unet(params)

    if params.model == 'volumeNet':
        model = volumeNet.unet(params)

    '''Compile model'''
    model.compile(optimizer=adam(lr=params.learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy', sm.metrics.iou_score])

    if params.continue_training:
        print("loaded already trained model")
        model.load_weights(params.pretrained_model + "/weights.hdf5", by_name=True, skip_mismatch=True)

    model.summary()
    return model
