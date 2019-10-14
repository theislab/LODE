#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import labelme


def main():
    labels_file = "labels.txt"
    in_dir = "./topcon_images"
    out_dir = "./data_dataset_voc"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    if osp.exists(out_dir):
        print('Output directory already exists:', out_dir)
        quit(1)
    os.makedirs(out_dir)
    os.makedirs(osp.join(out_dir, 'JPEGImages'))
    os.makedirs(osp.join(out_dir, 'SegmentationClass'))
    os.makedirs(osp.join(out_dir, 'SegmentationClassPNG'))
    os.makedirs(osp.join(out_dir, 'SegmentationClassVisualization'))
    os.makedirs(osp.join(out_dir, 'SegmentationObject'))
    os.makedirs(osp.join(out_dir, 'SegmentationObjectPNG'))
    os.makedirs(osp.join(out_dir, 'SegmentationObjectVisualization'))
    print('Creating dataset:', out_dir)

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
    out_class_names_file = osp.join(out_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    for label_file in glob.glob(osp.join(in_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                out_dir, 'JPEGImages', base + '.jpg')
            out_cls_file = osp.join(
                out_dir, 'SegmentationClass', base + '.npy')
            out_clsp_file = osp.join(
                out_dir, 'SegmentationClassPNG', base + '.png')
            out_clsv_file = osp.join(
                out_dir, 'SegmentationClassVisualization', base + '.jpg')
            out_ins_file = osp.join(
                out_dir, 'SegmentationObject', base + '.npy')
            out_insp_file = osp.join(
                out_dir, 'SegmentationObjectPNG', base + '.png')
            out_insv_file = osp.join(
                out_dir, 'SegmentationObjectVisualization', base + '.jpg')

            data = json.load(f)

            def to_binary(data):
                for i in range(0, len(data["shapes"])):
                    if data["shapes"][i]["label"] == "10":
                        data["shapes"][i]["label"] = "_background_"
                    else:
                        data["shapes"][i]["label"] = "1"
                return(data)

            data = to_binary(data)
            #data["imagePath"] = data["imagePath"].replace("..\\TODO\\", "").replace("..\\","").replace("Desktop\\","")
            #data["imagePath"] = os.path.join("/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/data/clinic_data/longitudinal_img_",
            #                                 data['imagePath'].split("/")[-1])
            #img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            #img = np.asarray(PIL.Image.open(img_file))
            #PIL.Image.fromarray(img).save(out_img_file)
            try:
                cls, ins = labelme.utils.shapes_to_label(
                    img_shape=(496,512),
                    shapes=data['shapes'],
                    label_name_to_value=class_name_to_id,
                    type='instance',
                )
                ins[cls == -1] = 0  # ignore it.

                #clsv = labelme.utils.draw_label(
                #    cls, img, class_names, colormap=colormap)
                #PIL.Image.fromarray(clsv).save(out_clsv_file)
                # class label
                #labelme.utils.lblsave(out_clsp_file, cls)
                cls_stack = np.stack((cls,) * 3, axis=-1).astype(np.uint8) * 255
                cv2.imwrite(out_cls_file.replace(".npy", ".png"), cls_stack)
                # instance label
            except:
                print("record not working is: {}".format(data["imagePath"]))
                continue


if __name__ == '__main__':
    main()
