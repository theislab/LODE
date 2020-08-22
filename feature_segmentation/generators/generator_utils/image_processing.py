import numpy as np
from PIL import Image


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
    image = im_resized.reshape(shape[0], shape[1], 3)
    label = lbl_resized.reshape((shape[0], shape[1], 1))
    return image, label