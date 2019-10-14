import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
label_path = "/home/olle/PycharmProjects/thickness_map_prediction/\
retinal_thickness_segmentation/data/clinic_data/test_labels"
labels = [os.path.join(label_path,i) for i in os.listdir(label_path)]
k = 7
for i in range(0,k*2):
    img = np.array(Image.open(labels[i]))