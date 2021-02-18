import os
import sys
from pathlib import Path

from .advanced_unets.models import att_unet

path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(path_variable, "networks"))
sys.path.insert(0, str(path_variable.parent))
sys.path.insert(0, str(path_variable.parent.parent))

from .networks import standard_unet, deep_unet, SEdeep_unet, deeper_unet, volumeNet, \
    cluster_unet, deep_se_unet, pyramid_unet, pyramid_transpose_unet, pyramid_deep_unet, \
    deep_s_class_e_unet, deep_aline_attention_unet, deep_attention_unet_avg_pooling,standard_unet_aa


def get_model(params):

    available_models = ["attention_unet",
                        "deep_unet",
                        "deep_se_unet",
                        "deep_s_class_e_unet",
                        "pyramid_unet",
                        "pyramid_transpose_unet",
                        "pyramid_deep_unet",
                        "deep_aline_attention_unet",
                        "standard_unet_aa",
                        "standard_unet",
                        "deep_attention_unet_avg_pooling"]

    assert params.model in available_models, f"model not available, choose from {available_models}"

    if params.model == 'standard_unet':
        model = standard_unet.unet(params)

    if params.model == 'standard_unet_aa':
        model = standard_unet_aa.unet(params)

    if params.model == 'deep_unet':
        model = deep_unet.unet(params)

    if params.model == 'deep_se_unet':
        model = deep_se_unet.unet(params)

    if params.model == 'deep_s_class_e_unet':
        model = deep_s_class_e_unet.unet(params)

    if params.model == 'deep_aline_attention_unet':
        model = deep_aline_attention_unet.unet(params)

    if params.model == 'deeper_unet':
        model = deeper_unet.unet(params)

    if params.model == 'deep_attention_unet_avg_pooling':
        model = deep_attention_unet_avg_pooling.unet(params)

    if params.model == 'SEdeep_unet':
        model = SEdeep_unet.unet(params)

    if params.model == 'volumeNet':
        model = volumeNet.unet(params)

    if params.model == 'pyramid_unet':
        model = pyramid_unet.unet(params)

    if params.model == 'pyramid_transpose_unet':
        model = pyramid_transpose_unet.unet(params)

    if params.model == 'pyramid_deep_unet':
        model = pyramid_deep_unet.unet(params)

    if params.model == "cluster_unet":
        model = cluster_unet.unet(params)

    if params.model == "attention_unet":
        model = att_unet(params, data_format = 'channels_last')

    if params.continue_training:
        print("loaded already trained model")
        model.load_weights(params.pretrained_model + "/weights.hdf5", by_name=True, skip_mismatch=True)

    model.summary()
    return model
