import numpy as np
from PIL import Image
import cv2
from random import random;


def invert_camera_effect(img):
    # annotate camera artifact
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = img
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    img[mask[:, :] == 0] = 255

    return np.stack((img,) * 3, axis = -1)


def resize(im, shape):
    desired_size = shape[0]
    im = Image.fromarray(im)

    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.NEAREST)
    # create a new image and paste the resized on it

    new_im = Image.new("L", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))
    return np.array(new_im)


def read_resize_random_invert(img_path, label_path, shape):
    # load samples
    im = Image.open(img_path)
    lbl = Image.open(label_path)

    # convert to numpy
    im = np.array(im)
    lbl = np.array(lbl)

    if random() < 0.2:
        im = invert_camera_effect(im)

    # resize samples
    im_resized = resize(im, shape)
    lbl_resized = resize(lbl, shape)

    # if image grey scale, make 3 channel
    if len(im_resized.shape) == 2:
        im_resized = np.stack((im_resized,) * 3, axis = -1)

    # Store sample
    image = im_resized.reshape(shape[0], shape[1], 3)
    label = lbl_resized.reshape((shape[0], shape[1], 1))
    return image, label


def read_resize(img_path, label_path, shape):
    # load samples
    im = Image.open(img_path)
    lbl = Image.open(label_path)

    # convert to numpy
    im = np.array(im)
    lbl = np.array(lbl)

    # resize samples
    im_resized = resize(im, shape)
    lbl_resized = resize(lbl, shape)

    # if image grey scale, make 3 channel
    if len(im_resized.shape) == 2:
        im_resized = np.stack((im_resized,) * 3, axis = -1)

    # Store sample
    image = im_resized.reshape((shape[0], shape[1], 3))
    label = lbl_resized.reshape((shape[0], shape[1], 1))
    return image, label


def read_resize_image(img_path, shape):
    # load samples
    im = Image.open(img_path)

    # convert to numpy
    im = np.array(im)

    # resize samples
    im_resized = resize(im, shape)

    # if image grey scale, make 3 channel
    if len(im_resized.shape) == 2:
        im_resized = np.stack((im_resized,) * 3, axis = -1)

    # Store sample
    image = im_resized.reshape(shape[0], shape[1], 3)
    return image