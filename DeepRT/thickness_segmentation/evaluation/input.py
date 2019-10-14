import functools
import tensorflow as tf
from params import params
import math

img_shape = params["img_shape"]
batch_size = params["batch_size"]
sess = tf.InteractiveSession()
def _process_pathnames(fname, label_path):
    '''
    Read in images in jpeg,png or gif formats as bytes for tf.data pipeline
    :param fname: path of images
    :param label_path: path to labels
    :return: images and labels as bytes
    '''
    # We map this function onto each pathname pair
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    label_img_str = tf.read_file(label_path)
    # These are gif images so they return as (num_frames, h, w, c)
    label_img = tf.image.decode_jpeg(label_img_str)[0]
    # The label image should only have values of 1 or 0, indicating pixel wise
    # object (car) or not (background). We take the first channel only.
    label_img = tf.expand_dims(label_img, axis=-1)
    return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([],
                                                  -width_shift_range * img_shape[1],
                                                  width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                                   -height_shift_range * img_shape[0],
                                                   height_shift_range * img_shape[0])
        # Translate both
        output_img = tf.contrib.image.translate(output_img,
                                               [width_shift_range, height_shift_range])
        label_img = tf.contrib.image.translate(label_img,
                                              [width_shift_range, height_shift_range])
    return output_img, label_img


def rotate_images(output_img, label_img):
    degree_angle = tf.random_uniform([],minval= -90 ,maxval= 90)

    radian = degree_angle * math.pi / 180

    img_rotate = tf.contrib.image.rotate(output_img, radian)
    label_rotate = tf.contrib.image.rotate(label_img, radian)

    return img_rotate, label_rotate

def flip_img(horizontal_flip, tr_img, label_img):
  if horizontal_flip:
    flip_prob = tf.random_uniform([], 0.0, 1.0)
    tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                lambda: (tr_img, label_img))
  return tr_img, label_img


def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0,
             rotate=False):  # Randomly translate the image vertically
    if resize is not None:
        # Resize both images
        label_img = tf.image.resize_images(label_img, resize)
        img = tf.image.resize_images(img, resize)

    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)
    if rotate:
        img,label_img = rotate_images(img,label_img)

    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)

    label_img = tf.to_float(label_img) #* scale
    img = tf.to_float(img) * scale
    return img, label_img

tr_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
    'hue_delta': 0.1,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'rotate': True
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

val_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5,
                         batch_size=batch_size,
                         shuffle=True):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat().batch(batch_size)
    return dataset

def batch_data_sets(x_train_filenames,y_train_filenames,x_val_filenames,y_val_filenames):

    train_ds = get_baseline_dataset(x_train_filenames,
                                    y_train_filenames,
                                    preproc_fn=tr_preprocessing_fn,
                                    batch_size=params["batch_size"])
    val_ds = get_baseline_dataset(x_val_filenames,
                                  y_val_filenames,
                                  preproc_fn=val_preprocessing_fn,
                                  batch_size=params["batch_size"])

    temp_ds = get_baseline_dataset(x_train_filenames,
                                   y_train_filenames,
                                   preproc_fn=tr_preprocessing_fn,
                                   batch_size=1,
                                   shuffle=False)
    return(train_ds,val_ds,temp_ds)