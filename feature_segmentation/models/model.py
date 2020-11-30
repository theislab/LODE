import os
import sys
from pathlib import Path

from .advanced_unets.models import att_unet

path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(path_variable, "networks"))
sys.path.insert(0, str(path_variable.parent))
sys.path.insert(0, str(path_variable.parent.parent))

from .networks import standard_unet, deep_unet, SEdeep_unet, deeper_unet, volumeNet, cluster_unet


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

    if params.model == "cluster_unet":
        model = cluster_unet.unet(params)

    if params.model == "attention_unet":
        model = att_unet(params, data_format = 'channels_last')

    if params.continue_training:
        print("loaded already trained model")
        model.load_weights(params.pretrained_model + "/weights.hdf5", by_name=True, skip_mismatch=True)

    model.summary()
    return model
