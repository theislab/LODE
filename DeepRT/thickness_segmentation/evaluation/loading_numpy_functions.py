import random
from PIL import Image
from PIL import ImageOps
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import os
import random
import skimage as sk
from skimage import transform
import cv2
import matplotlib.pyplot as plt
from params import params
import regex as re
from numpy import inf
def brightness_augment(img, factor=0.5):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.float32), cv2.COLOR_HSV2RGB)
    return rgb

def random_rotation(image_array, label_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    r_im = sk.transform.rotate(image_array, random_degree, preserve_range = True)
    r_l = sk.transform.rotate(label_array, random_degree, preserve_range = True)
    return r_im.astype(np.float32), r_l.astype(np.uint8)

def elastic_transform(image, mask, alpha=720, sigma=24, alpha_affine=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))


    res_x = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    res_y = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    return res_x, res_y

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
    result = np.zeros([int(new_shape[0]), int(new_shape[1])])
    #print(new_shape[0],new_shape[1], orig_shape[0], orig_shape[1])
    x_offset = int((new_shape[0] - orig_shape[0])/2)  # 0 would be what you wanted
    y_offset = int((new_shape[1] - orig_shape[1])/2)  # 0 in your case
    #print(x_offset, y_offset)
    
    result[x_offset:im.shape[0]+x_offset,y_offset:im.shape[1]+y_offset] = im
    return(result)


def fill_label_gaps(seg_im):

    for i in range(0, seg_im.shape[1]):
        try:
            max_one = np.max(np.argwhere(seg_im[:, i] == 1))
            min_one = np.min(np.argwhere(seg_im[:, i] == 1))
            seg_im[min_one:max_one, i] = 1
        except:
            continue
    return (seg_im)

def eval_lod(dir, im_name):
    size = (params["img_shape"][0], params["img_shape"][1])
    train_im = cv2.imread(os.path.join(dir, "image_sources/topcon_images",str(im_name)+".png"))  # .convert('L')
    im_resized = cv2.resize(train_im, (256,256))
    # scale image
    im_resized = np.divide(im_resized, 255, dtype=np.float32)

    seg_im = cv2.imread(os.path.join(dir, "image_sources/topcon_labels",str(im_name)+".png"))  # .convert('L')
    # duke data have 2's appearing, remove
    seg_im[seg_im == 2] = 0
    seg_im = fill_label_gaps(seg_im)

    # im = np.array(im)
    seg_resized = cv2.resize(seg_im,size) / 255

    im_batch = np.reshape(np.asarray(im_resized),
                          (1, params["img_shape"][0], params["img_shape"][1], 3))
    labels_batch = np.reshape(np.asarray(seg_resized[:,:,0], dtype = np.int32),
                              (1, params["img_shape"][0], params["img_shape"][1], 1))
    return  im_batch , labels_batch
def get_clinic_train_data(im_dir, seg_dir,  img_shape,batch_size):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''
    img_width = img_shape[0]
    img_height = img_shape[1]
    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
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
        # random int for selecting images
        random_int = [0]#np.random.choice(len(im_names), batch_size)

        im_name = im_names[random_int[0]]
        im_displayed.append(im_name)
        seg_name = [s for s in seg_names if im_name.replace(".jpeg","").replace(".dcm","") in s]
        if batch_size > k:
            k += 1
            # retrieve image
            train_im = Image.open(os.path.join(im_dir, im_name))#.convert('L')
            train_im = np.array(train_im)

            im_resized = Image.fromarray(train_im).resize(size, Image.NEAREST)

            # scale image
            im_resized = np.divide(im_resized, 255, dtype=np.float32)

            seg_im = Image.open(os.path.join(seg_dir,seg_name[0]))#.convert('L')
            seg_im = np.array(seg_im)
            #duke data have 2's appearing, remove
            seg_im[seg_im == 2] = 0
            seg_im = fill_label_gaps(seg_im)


            # im = np.array(im)
            seg_resized = Image.fromarray(seg_im).resize(size)

            # get random numbers to determine whether to do data augmentation or not
            float_horizontal_flip = 0#random.uniform(0.0, 1.0)
            float_vertical_flip = 0#random.uniform(0.0, 1.0)
            float_elastic_tranform = 0#random.uniform(0.0, 1.0)
            float_rotation = random.uniform(0.0, 1.0)
            float_brightness = random.uniform(0.0, 1.0)

            if float_brightness > 0.5:
                im_resized = brightness_augment(im_resized)
            # 50 % chance that both im and seg is flipped
            if float_horizontal_flip > 0.5:
                im_resized = np.fliplr(im_resized)
                seg_resized = np.fliplr(seg_resized)
            im_resized = np.array(im_resized)
            seg_resized = np.array(seg_resized, dtype=np.uint8)
            #
            if float_vertical_flip > 0.5:
                im_resized = np.flip(im_resized, 0)
                seg_resized = np.flip(seg_resized, 0)
            im_resized = np.array(im_resized)
            seg_resized = np.array(seg_resized, dtype=np.uint8)

            if float_elastic_tranform > 0.5:
                im_resized, seg_resized = elastic_transform(im_resized, seg_resized)

            if float_rotation > 0.5:
                im_resized, seg_resized = random_rotation(im_resized, seg_resized)

            #invert the images with reversed colors
            if np.mean(im_resized) > 90:
                im_resized = np.invert(im_resized)

            #if image is grey scale then make into 3 color channel
            if len(im_resized.shape) != 3:
                im_resized = cv2.cvtColor(im_resized.astype(np.float32),cv2.COLOR_GRAY2RGB)
            images.append(im_resized)
            seg_maps.append(seg_resized)

    im_batch = np.reshape(np.asarray(images), (batch_size, img_height, img_width, 3))
    labels_batch = np.reshape(np.asarray(seg_maps, dtype = np.int32), (batch_size, img_height, img_width, 1))

    return (im_batch, labels_batch, im_displayed)