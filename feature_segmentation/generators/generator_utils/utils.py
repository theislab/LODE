import os
import numpy as np
import itertools
from collections import Counter
from PIL import Image


def get_class_distribution(lbl_path, ids):
    full_lbl_paths = [os.path.join(lbl_path, id) for id in ids]
    label_repr = [np.unique(Image.open(lp)).tolist() for lp in full_lbl_paths]
    flatten = list(itertools.chain(*label_repr))
    class_distribution = Counter(flatten)

    upsampling_factors = {}
    for key in class_distribution.keys():
        upsampling_factors[key] = len(ids) // class_distribution[key]
    return upsampling_factors, label_repr


def upsample(ids, lbl, lr, uf):
    new_ids = []
    for k, id in enumerate(ids):
        if lbl in lr[k]:
            new_ids = new_ids + [id] * uf[lbl]
    return new_ids + ids