import numpy as np
import os
import random
from PIL import Image
from PIL import ImageOps
import regex as re
import cv2
def get_unique_string(im_name):
    '''
    :param im_name: a string with the image file name
    :return: the unique identifier for that image
    '''
    im_parts = im_name.split("_")

    strings = []
    for parts in im_parts:
        if re.findall('\d+', parts):
            strings.append(re.findall('\d+', parts))

    unique_str = strings[0][0] + "_" + strings[1][0]
    return (unique_str)

def padding_with_zeros(im, orig_shape, new_shape):
    '''
    :param im:
    :param orig_shape:
    :param new_shape:
    :return:
    '''
#    im = im.reshape(orig_shape)
    result = np.zeros((int(new_shape[0]), int(new_shape[1])))
    #print(new_shape[0],new_shape[1], orig_shape[0], orig_shape[1])
    x_offset = int((new_shape[0] - orig_shape[0])/2)  # 0 would be what you wanted
    y_offset = int((new_shape[1] - orig_shape[1])/2)  # 0 in your case
    #print(x_offset, y_offset)
    
    result[x_offset:im.shape[0]+x_offset,y_offset:im.shape[1]+y_offset] = im
    return(result)

def get_clinic_train_data(im_dir, seg_dir, img_width, img_height,batch_size):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''

    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
    # random int for selecting images
    random_int = np.random.choice(len(im_names),batch_size)
    # random_int = 1
    # set containers holding data
    images = []
    seg_maps = []
    im_id = []
    im_displayed = []
    # set sizes
    size = [img_width, img_height]
    # gather data
    k = 0
    for i in range(batch_size):
        im_name = im_names[random_int[i]]
        unique_str = get_unique_string(im_name)
        im_displayed.append(unique_str)
        # print("Just feeding same image")

        if batch_size > k:
            k += 1
            # retrieve image
            train_im = Image.open(im_dir + im_name).convert('L')
            train_im = np.array(train_im)
            orig_shape = [train_im.shape[0], train_im.shape[1]]
            new_shape = [train_im.shape[0], train_im.shape[1] * np.divide(img_width, img_height)]
            #print(new_shape, orig_shape)
            # print(train_im.shape)
            im_padded = padding_with_zeros(train_im, orig_shape, new_shape)
            # im = np.array(im)
            im_resized = Image.fromarray(im_padded).resize(size, Image.NEAREST)

            # retrieve the labels
            print("The unique string is {}".format(unique_str))
            y_path = [s for s in seg_names if unique_str in s]
            seg_im = Image.open(seg_dir + y_path[0])  # .convert('L')
            seg_im = np.array(seg_im)
            seg_padded = padding_with_zeros(seg_im, orig_shape, new_shape)
            # im = np.array(im)
            seg_resized = Image.fromarray(seg_padded).resize(size)

            float_r = random.uniform(0.0, 1.0)
            # 50 % chance that both im and seg is flipped
            if float_r > 0.5:
                im_resized = ImageOps.mirror(im_resized)
                seg_resized = ImageOps.mirror(seg_resized)

            im_resized = np.array(im_resized)
            seg_resized = np.array(seg_resized, dtype = np.uint8)

            #invert the images with reversed colors
            if np.mean(im_resized) > 100:
                im_resized = np.invert(im_resized)

            images.append(im_resized)
            seg_maps.append(seg_resized)

    im_batch = np.reshape(np.asarray(images), (batch_size, 160, 400, 1))
    labels_batch = np.reshape(np.asarray(seg_maps, dtype = np.int32), (batch_size, 160, 400, 1))

    return (im_batch, labels_batch, im_displayed)

def get_clinic_data_hardrive(im_dir, img_width, img_height):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''
    size = [img_width, img_height]
    train_im = cv2.imread(im_dir,0)
    orig_shape = [train_im.shape[0], train_im.shape[1]]
    new_shape = [train_im.shape[0], train_im.shape[1] * np.divide(img_width, img_height)]
    # print(train_im.shape)
    im_padded = padding_with_zeros(train_im, orig_shape, new_shape)
    # im = np.array(im)
    im_resized = Image.fromarray(im_padded).resize(size, Image.NEAREST)
    im_resized = np.array(im_resized, dtype = np.uint8)
    #invert the images with reversed colors
    if np.mean(im_resized) > 90:
        im_resized = np.invert(im_resized)
    ###IMAGE READY FOR PREDICTION
    im_batch = np.reshape(im_resized, (1, 160, 400, 1))
    return im_batch, new_shape, orig_shape