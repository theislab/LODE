from PIL import Image
import numpy as np


def resize(im, size):
    """
    :param im: image to be resized
    :type im: numpy array
    :param size: tuple with (width, height, channels)
    :type size: tuple
    :return: resized array
    :rtype: array
    """
    desired_size = size
    im = Image.fromarray(im)
    old_size = im.size

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.NEAREST)

    # create a new image and paste the resized on it
    new_im = Image.new("L", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))

    return np.array(new_im)
