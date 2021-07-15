import tensorflow.keras

from models.networks import deep_unet


def get_model(params):
    model = deep_unet.unet(params)
    if params.continue_training:
        print("loaded already trained model")
        model = tensorflow.keras.models.load_model(params.pretrained_model + "/weights.hdf5")

    model.summary()
    return model
