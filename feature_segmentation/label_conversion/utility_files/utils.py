from __future__ import print_function
import math
import os
import os.path as osp
import numpy as np
from PIL import ImageDraw, Image
from utility_files.fibrosis_record_changes import map_changes
from utility_files.smoothing import smooth_cubic_interp
import matplotlib.pyplot as plt

def change_string_label_to_integer(shapes):
    for elem in shapes:
        if str(elem['label']) == 'epiretinal membrane':
            elem['label'] = '1'
        if str(elem['label']) == 'neurosensory retina':
            elem['label'] = '2'
        if str(elem['label']) == 'intraretinal fluid':
            elem['label'] = '3'
        if str(elem['label']) == 'subretinal fluid':
            elem['label'] = '4'
        if (str(elem['label']) == 'subretinal hyper reflective material') or \
                (str(elem['label']) == 'subretinalhyper reflective material'):
            elem['label'] = '5'
        if str(elem['label']) == 'RPE':
            elem['label'] = '6'
        if str(elem['label']) == 'fibrovascular PED':
            elem['label'] = '7'
    return shapes


def mask_overlap(mask, masks_history):
    overlap = []
    for mask_ in masks_history:
        if np.sum(mask & mask_[0]) > 0:
            overlap.append(1)
        else:
            overlap.append(0)
    return overlap


def set_segmap(cls, mask, cls_id, masks_history):
    # set new mask
    cls[mask] = cls_id

    # calculate size of polygon
    mask_size = np.sum(mask)

    # get overlap with all other masks made
    overlaps = mask_overlap(mask, masks_history)

    for k, overlap in enumerate(overlaps):
        if overlap == 1:
            overlapped_mask = masks_history[k][0]
            overlapped_mask_size = np.sum(overlapped_mask)

            if overlapped_mask_size < mask_size:
                cls[overlapped_mask] = masks_history[k][1]


# clean data image path
def clean_data_path(image_path):
    image_path = image_path.split("\\")[-1]
    image_path = image_path.split("/")[-1]
    image_path = image_path.replace("jpeg", "png")
    image_path = image_path.replace(".dcm", "")
    return image_path


def assert_image_label_files(label_file, image_file):
    l_name = label_file.split("/")[-1].replace(".json", "")
    i_name = image_file.split("/")[-1].replace(".dcm.jpeg", "")
    if i_name != l_name:
        i_name = l_name
    return i_name + ".dcm.jpeg"


def set_outdir(out_dir, labels_file):
    # create annotator specific outdir
    os.makedirs(out_dir, exist_ok = True)
    os.makedirs(osp.join(out_dir, 'images'), exist_ok = True)
    os.makedirs(osp.join(out_dir, 'masks'), exist_ok = True)
    os.makedirs(osp.join(out_dir, 'visualizations_pre_processing'), exist_ok = True)
    os.makedirs(osp.join(out_dir, 'visualizations_post_processing'), exist_ok = True)
    os.makedirs(osp.join(out_dir, 'overview'), exist_ok = True)
    os.makedirs(osp.join(out_dir, 'json'), exist_ok = True)

    # load label names from txt file
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels_file).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)

    # write labels txt to outdir
    out_class_names_file = osp.join(out_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)
    return class_name_to_id


def set_image(image_path):
    return (image_path.replace("/", "").replace("TODO", "").replace("DONE", "") \
            .replace("JSON", "").replace("./all_annotated_iter2/..", "").replace("..", ""))


def iter_one_processing(cls):
    # set geographic atrophy to background
    cls[cls == 8] = 0

    # set drusen 9 to 8, and posterios hylaoid membrane 10 to 9
    cls[cls == 9] = 8
    cls[cls == 10] = 9
    return cls


def shapes_to_label(img_shape, shapes, label_name_to_value, type='class', smoothen=False):
    masks_history = []
    # change all string labels to character
    shapes = change_string_label_to_integer(shapes)
    assert type in ['class', 'instance']
    # order list of dictionaries
    shapes_ordered = sorted(shapes, key = lambda k: k['label'])
    cls = np.zeros(img_shape[:2], dtype = np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype = np.int32)
        instance_names = ['_background_']
    for shape in shapes_ordered:
        points = shape['points']
        if smoothen:
            points = smooth_cubic_interp(points)  # smooth_points(points)
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)
            ins_id = len(instance_names) - 1
        cls_id = label_name_to_value[cls_name]

        if cls_id != 10:
            shape_type = "polygon"
        mask = shape_to_mask(img_shape[:2], points, shape_type)

        set_segmap(cls, mask, cls_id, masks_history)
        masks_history.append([mask, cls_id])

        if type == 'instance':
            ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls


def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype = np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline = 1, fill = 1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline = 1, fill = 1)
    elif shape_type == 'line':
        # assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy = xy, fill = 1, width = line_width)
    elif shape_type == 'linestrip':
        draw.line(xy = xy, fill = 1, width = line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline = 1, fill = 1)
    else:
        try:
            # assert len(xy) > 2, 'Polygon must have points more than 2'
            draw.polygon(xy = xy, outline = 1, fill = 1)
        except:
            print("polygon length is: {}".format(len(xy)))
    mask = np.array(mask, dtype = bool)
    return mask


def fibrosis_change(img_id, change_log, map_):
    record_bool = change_log["id"] == img_id
    if np.sum(record_bool) != 0:
        action = change_log.loc[record_bool]["action"].values[0]
        if action == "change":
            map_[map_ == 5] = 13
        elif action == "dont change":
            return map_
        elif action == "change custom":
            return map_changes(img_id, map_)
    else:
        return map_
    return map_
