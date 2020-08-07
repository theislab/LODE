from copy import deepcopy
import collections.abc

def merged_config(config1, config2):
    """update config1 with config2. Should work arbitary nested levels"""
    res_config = deepcopy(config1)
    for k, v in config2.items():
        if isinstance(v, collections.abc.Mapping):
            res_config[k] = merged_config(config1.get(k, {}), v)
        else:
            res_config[k] = v
    return res_config