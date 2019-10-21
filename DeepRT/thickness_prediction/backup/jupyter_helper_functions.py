import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import gc
from scipy import ndimage


path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/normal_range_stats"
sd_oct_label = "_label_Macular_Thickness_by_Age_and_Gender_in_Healthy_Eyes_Using_Spectral_Domain_Optical_\
Coherence_Tomography.csv"
sd_oct_prediction = "_prediction_Macular_Thickness_by_Age_and_Gender_in_Healthy_Eyes_Using\
_Spectral_Domain_Optical_Coherence_Tomography.csv"

label_path = "/home/olle/PycharmProjects/thickness_map_prediction/\
project_evaluation/test_labels/"
prediction_path = "/home/olle/PycharmProjects/thickness_map_prediction/\
project_evaluation/test_predictions"
image_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/test_images"

def load_images(record_name):
    label = np.load(os.path.join(label_path, record_name))
    prediction = np.load(os.path.join(prediction_path, record_name))
    image = cv2.imread(os.path.join(image_path, record_name.replace(".npy",".jpeg")))

	
    label_mu = pixel_to_mu_meter(label)
    prediction_mu = pixel_to_mu_meter(prediction)
    # resize prediciton
    prediction_mu = resize_prediction(prediction_mu)

    # remove three channel to stop normalization
    label_lr = get_low_res_depth_grid(label_mu)[:, :, 0]
    prediction_lr = get_low_res_depth_grid(prediction_mu)[:, :, 0]
    return (label_mu, prediction_mu, label_lr, prediction_lr,image)

def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_zone(record_th_z, zone):
    zone_value = record_th_z[record_th_z.Name == zone].AvgThickness.iloc[0]
    # mean of all zone thickness
    zone_avg = np.nanmean(np.array(record_th_z.AvgThickness, dtype=np.float32))

    if zone_value is None:
        zone_value = zone_avg

    return (float(zone_value))


def extract_values(record_th_z):
    C0_value = get_zone(record_th_z, "C0")
    S2_value = get_zone(record_th_z, "S2")
    S1_value = get_zone(record_th_z, "S1")
    N1_value = get_zone(record_th_z, "N1")
    N2_value = get_zone(record_th_z, "N2")
    I1_value = get_zone(record_th_z, "I1")
    I2_value = get_zone(record_th_z, "I2")
    T1_value = get_zone(record_th_z, "T1")
    T2_value = get_zone(record_th_z, "T2")
    return (C0_value, S2_value, S1_value, N1_value, N2_value, I1_value, I2_value, T1_value, T2_value)


def set_low_res_depth_grid(C0_value, S2_value, S1_value, N1_value, N2_value, I1_value, I2_value, T1_value, T2_value,
                           C0, S2, S1, N1, N2, I1, I2, T1, T2, img):
    img[C0] = C0_value
    img[S1] = S1_value
    img[S2] = S2_value
    img[I1] = I1_value
    img[I2] = I2_value
    img[T1] = T1_value
    img[T2] = T2_value
    img[N1] = N1_value
    img[N2] = N2_value

    return (img)

def rescale_oct_height(depth_map):
    scaling_factor = np.divide(496,160,dtype=np.float32)
    rescaled_depth_map = depth_map * scaling_factor
    return(rescaled_depth_map)

def center_img(img):
    coords = np.argwhere(img > 0)
    x_min, y_min = coords.min(axis=0)[0:2]
    x_max, y_max = coords.max(axis=0)[0:2]
    cropped_img = img[x_min:x_max - 1, y_min:y_max - 1]
    if len(cropped_img.shape) == 2:
        square_cropped_img = cropped_img[0:min(cropped_img.shape),0:min(cropped_img.shape)]

        centered_img = np.zeros((768, 768))

        nb = centered_img.shape[0]
        na = square_cropped_img.shape[0]
        lower = (nb) // 2 - (na // 2)
        upper = (nb // 2) + (na // 2)

        difference = np.abs(lower - upper) - square_cropped_img.shape[0]
        upper = upper - difference

        centered_img[lower:upper, lower:upper] = square_cropped_img

    if len(cropped_img.shape) == 3:
        square_cropped_img = cropped_img[0:min(cropped_img.shape[0:2]), 0:min(cropped_img.shape[0:2]), :]
        centered_img = np.zeros((768, 768,3)).astype(np.uint8)

        nb = centered_img.shape[0]
        na = square_cropped_img.shape[0]
        lower = (nb) // 2 - (na // 2)
        upper = (nb // 2) + (na // 2)

        difference = np.abs(lower - upper) - square_cropped_img.shape[0]
        upper = upper - difference

        centered_img[lower:upper, lower:upper,:] = square_cropped_img

    return(centered_img)


def get_low_res_grid(img):
    # scale of LOCALIZER
    outer_ring_radius = int(6 / 0.0118) / 2
    middle_ring_radius = int(3 / 0.0118) / 2
    inner_ring_radius = int(1 / 0.0118) / 2

    min_ = min(img.nonzero()[0]), min(img.nonzero()[1])
    max_ = max(img.nonzero()[0]), max(img.nonzero()[1])
    image_span = np.subtract(max_,min_)

    measure_area = np.zeros(image_span)

    nrows = 768
    ncols = 768
    cnt_row = image_span[1] / 2 + min_[1]
    cnt_col = image_span[0] / 2 + min_[0]

    max_diam = min(image_span)

    # init empty LOCALIZER sized grid
    img_mask = np.zeros((nrows, ncols), np.float32)

    # create base boolean masks
    inner_ring_mask = createCircularMask(nrows, ncols, center=(cnt_row, cnt_col), radius=inner_ring_radius)
    middle_ring_mask = createCircularMask(nrows, ncols, center=(cnt_row, cnt_col), radius=middle_ring_radius)

    #fit low res grid to measurement area
    if outer_ring_radius*2 > max_diam:
        outer_ring_radius = max_diam/2

    outer_ring_mask = createCircularMask(nrows, ncols, center=(cnt_row, cnt_col), radius=outer_ring_radius)

    inner_disk = inner_ring_mask
    middle_disk = (middle_ring_mask.astype(int) - inner_ring_mask.astype(int)).astype(bool)
    outer_disk = (outer_ring_mask.astype(int) - middle_ring_mask.astype(int)).astype(bool)

    #create label specific diagonal masks
    upper_triangel_right_mask = np.arange(0,img.shape[1])[:, None] <= np.arange(img.shape[1])
    lower_triangel_left_mask = np.arange(0,img.shape[1])[:, None] > np.arange(img.shape[1])
    upper_triangel_left_mask = lower_triangel_left_mask[::-1]
    lower_triangel_right_mask = upper_triangel_right_mask[::-1]
    ''''
    #pad the shortened arrays
    im_utr = np.zeros((768,768))
    im_ltl = np.zeros((768,768))
    im_utl = np.zeros((768,768))
    im_ltr = np.zeros((768,768))

    #pad the diagonal masks
    im_utr[0:upper_triangel_right_mask.shape[0],:] = upper_triangel_right_mask
    im_ltl[768-upper_triangel_right_mask.shape[0]:,:] = lower_triangel_left_mask
    im_utl[0:upper_triangel_left_mask.shape[0], :] = upper_triangel_left_mask
    im_ltr[768-lower_triangel_right_mask.shape[0]:, :] = lower_triangel_right_mask
    #conversion
    im_utr = im_utr.astype(np.bool)
    im_ltl = im_ltl.astype(np.bool)
    im_utl = im_utl.astype(np.bool)
    im_ltr = im_ltr.astype(np.bool)
    '''
    # create 9 depth regions
    C0 = inner_disk = inner_ring_mask
    S2 = np.asarray(upper_triangel_left_mask & outer_disk & upper_triangel_right_mask)
    S1 = np.asarray(upper_triangel_left_mask & middle_disk & upper_triangel_right_mask)
    N1 = np.asarray(lower_triangel_right_mask & middle_disk & upper_triangel_right_mask)
    N2 = np.asarray(lower_triangel_right_mask & outer_disk & upper_triangel_right_mask)
    I1 = np.asarray(lower_triangel_right_mask & middle_disk & lower_triangel_left_mask)
    I2 = np.asarray(lower_triangel_right_mask & outer_disk & lower_triangel_left_mask)
    T1 = np.asarray(upper_triangel_left_mask & middle_disk & lower_triangel_left_mask)
    T2 = np.asarray(upper_triangel_left_mask & outer_disk & lower_triangel_left_mask)
    return(C0, S2, S1, N1, N2, I1, I2, T1, T2)

def get_depth_grid_edges(area):
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(area, struct)
    edges = area ^ erode
    return np.stack((edges,)*3, axis=-1)

def get_low_res_grid_shape(img):
    C0, S2, S1, N1, N2, I1, I2, T1, T2 = get_low_res_grid(img)
    grid = np.zeros((img.shape[0],img.shape[1],3), np.float32)
    grid = grid + get_depth_grid_edges(C0)
    grid = grid + get_depth_grid_edges(S1)
    grid = grid + get_depth_grid_edges(S2)
    grid = grid + get_depth_grid_edges(I1)
    grid = grid + get_depth_grid_edges(I2)
    grid = grid + get_depth_grid_edges(T1)
    grid = grid + get_depth_grid_edges(T2)
    grid = grid + get_depth_grid_edges(N1)
    grid = grid + get_depth_grid_edges(N2)
    return(grid)

def get_low_res_depth_grid(img):
    C0, S2, S1, N1, N2, I1, I2, T1, T2 = get_low_res_grid(img)
    grid = np.zeros((img.shape[0],img.shape[1],3), np.float32)
    grid[C0] = np.mean(img[C0])
    grid[S1] = np.mean(img[S1])
    grid[S2] = np.mean(img[S2])
    grid[I1] = np.mean(img[I1])
    grid[I2] = np.mean(img[I2])
    grid[T1] = np.mean(img[T1])
    grid[T2] = np.mean(img[T2])
    grid[N1] = np.mean(img[N1])
    grid[N2] = np.mean(img[N2])
    return(grid)

def pixel_to_mu_meter(img):
    img_um = np.multiply(img,0.0039*1000)
    return(img_um)
def get_low_res_depth_grid_values(img):
    C0, S1, S2, N1, N2, I1, I2, T1, T2 = get_low_res_grid(img)
    #turn zero to nan
    img[img < 10] = 0
    img[img == 0] = np.nan
    #get mean values
    C0_value = np.nanmean(img[C0])
    S1_value = np.nanmean(img[S1])
    S2_value = np.nanmean(img[S2])
    I1_value = np.nanmean(img[I1])
    I2_value = np.nanmean(img[I2])
    T1_value = np.nanmean(img[T1])
    T2_value = np.nanmean(img[T2])
    N1_value = np.nanmean(img[N1])
    N2_value = np.nanmean(img[N2])

    #concert back nan values to zero
    img = np.nan_to_num(img)

    low_grid_values = [C0_value, S1_value, S2_value, N1_value, N2_value, I1_value, I2_value, T1_value, T2_value]
    return(low_grid_values)


def get_text_coord(img):
    C0, S1,S2, N1, N2, I1, I2, T1, T2 = get_low_res_grid(img)

    S1_x_mc = np.median(np.where(S1 == True)[1])
    S1_y_mc = np.median(np.where(S1 == True)[0])

    S2_x_mc = np.median(np.where(S2 == True)[1])
    S2_y_mc = np.median(np.where(S2 == True)[0])

    N1_x_mc = np.median(np.where(N1 == True)[1])
    N1_y_mc = np.median(np.where(N1 == True)[0])

    N2_x_mc = np.median(np.where(N2 == True)[1])
    N2_y_mc = np.median(np.where(N2 == True)[0])

    I1_x_mc = np.median(np.where(I1 == True)[1])
    I1_y_mc = np.median(np.where(I1 == True)[0])

    I2_x_mc = np.median(np.where(I2 == True)[1])
    I2_y_mc = np.median(np.where(I2 == True)[0])

    T1_x_mc = np.median(np.where(T1 == True)[1])
    T1_y_mc = np.median(np.where(T1 == True)[0])

    T2_x_mc = np.median(np.where(T2 == True)[1])
    T2_y_mc = np.median(np.where(T2 == True)[0])

    C0_x_mc = S1_x_mc
    C0_y_mc = N2_y_mc

    coord_list = [C0_x_mc, C0_y_mc, S1_x_mc, S1_y_mc,S2_x_mc, S2_y_mc, \
                  N1_x_mc, N1_y_mc, N2_x_mc, N2_y_mc, I1_x_mc, I1_y_mc, I2_x_mc, I2_y_mc, \
                  T1_x_mc, T1_y_mc, T2_x_mc, T2_y_mc]
    return (coord_list)
def pixel_to_mu_meter(img):
    img_um = np.multiply(img,0.0039*1000)
    return(img_um)

def resize_prediction(img):
    prediction_resized = cv2.resize(img,(768,768))
    return(prediction_resized)
        
def write_depthgrid_values(coord_list,value_list):
    for i in range(0,int(len(coord_list)/2)):
        plt.text(coord_list[i*2], coord_list[(i+1)*2-1], str(int(value_list[i])), ha='center', va='center',
                 bbox=dict(facecolor='white'))



