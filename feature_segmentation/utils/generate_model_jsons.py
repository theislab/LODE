import os
import base64
import io
import json
import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
from scipy import ndimage
from skimage import measure
from skimage.measure import label, regionprops


def img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    img_bin = f.getvalue()
    if hasattr(base64, 'encodebytes'):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def get_label_label_list(mask, label):
    lll = []
    array = mask == label
    array = remove_small_blobs(array, label)

    array = array.astype(float)
    label_mask = measure.label(array)

    blob_labels = np.unique(label_mask)
    for blob in blob_labels:
        if blob != 0:
            blob_mask = label_mask == blob

            # fill out eventual holes
            blob_mask = ndimage.binary_fill_holes(blob_mask).astype(int)

            poly_points = get_label_polygon(mask=blob_mask)
            poly_points_sampled = sample_polygon_points(poly_points)

            lll.append({'label': str(label),
                        'line_color': None,
                        'fill_color': None,
                        'points': poly_points_sampled})
    return lll


def get_label_polygon(mask=None):
    from imantics import Mask
    polygons = Mask(mask).polygons()
    return polygons.points[0].tolist()


def sample_polygon_points(points):
    return points[0::3]

label_thresholds = {1: 50, 3: 50, 2: 250, 6: 100, 9: 50, 4:50}


def remove_small_blobs(mask, label_int):
    label_img = label(mask)
    rp = regionprops(label_img.astype(np.uint8))
    for prop in rp:
        if prop.area < label_thresholds[label_int]:
            label_img[label_img == prop.label] = 0

    # set all non zero regions to true
    label_img[label_img != 0] = 1
    return label_img


def create_model_json(record_name, mask, img):
    test_json = {'version': '3.6.12', 'flags': {}}

    label_list = []

    for label_int in [2, 3, 4, 6, 9, 1]:
        label_list = label_list + get_label_label_list(mask, label_int)

    test_json['shapes'] = label_list
    test_json['lineColor'] = [0, 255, 0, 128]
    test_json['fillColor'] = [255, 0, 0, 128]
    test_json['imagePath'] = record_name + ".png"
    test_json['imageData'] = img_arr_to_b64(img).decode('utf-8')
    test_json['imageHeight'] = img.shape[0]
    test_json['imageWidth'] = img.shape[1]
    return test_json


def save_model_json(test_json, save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, name), 'w') as fp:
        json.dump(test_json, fp, indent=4)
