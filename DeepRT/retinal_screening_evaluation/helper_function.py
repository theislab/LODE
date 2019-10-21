import os
import pandas as pd
import numpy as np
import cv2
def resize_prediction(img):
    prediction_resized = cv2.resize(img,(768,768))
    return(prediction_resized)
def get_record_name(test_path):
    pat_id = test_path.split("_")[0]
    study_date = test_path.split("_")[1]
    laterality = test_path.split("_")[2]
    series_time = test_path.split("_")[-1].replace(".npy","")
    return pat_id + "_" + study_date + "_" + laterality + "_" + series_time

def if_prediction(depth_map_path):
    return("predictions" in depth_map_path.split("/")[-1])

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

def get_low_res_grid():
    # scale of LOCALIZER
    scalexy = 0.0117
    outer_ring_radius = int(6 / 0.0117) / 2
    middle_ring_radius = int(3 / 0.0117) / 2
    inner_ring_radius = int(1 / 0.0117) / 2

    nrows = 768
    ncols = 768
    cnt_row = 768 / 2
    cnt_col = 768 / 2

    # init empty LOCALIZER sized grid
    img = np.zeros((nrows, ncols, 3), np.float32)

    # create base boolean masks
    inner_ring_mask = createCircularMask(nrows, ncols, center=(cnt_row, cnt_col), radius=inner_ring_radius)
    middle_ring_mask = createCircularMask(nrows, ncols, center=(cnt_row, cnt_col), radius=middle_ring_radius)
    outer_ring_mask = createCircularMask(nrows, ncols, center=(cnt_row, cnt_col), radius=outer_ring_radius)

    inner_disk = inner_ring_mask
    middle_disk = (middle_ring_mask.astype(int) - inner_ring_mask.astype(int)).astype(bool)
    outer_disk = (outer_ring_mask.astype(int) - middle_ring_mask.astype(int)).astype(bool)

    upper_triangel_right_mask = np.arange(img.shape[0])[:, None] <= np.arange(img.shape[1])
    lower_triangel_left_mask = np.arange(img.shape[0])[:, None] > np.arange(img.shape[1])
    upper_triangel_left_mask = lower_triangel_left_mask[::-1]
    lower_triangel_right_mask = upper_triangel_right_mask[::-1]

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
def scale_zero_one(img, scale_factor):
    img_scaled = img * scale_factor
    return(img_scaled)

def rescale_zero_one(img, scale_factor):
    img_rescaled = img / scale_factor
    return(img_rescaled)

def pixel_to_mu_meter(img):
    img_um = np.multiply(img,0.0039*1000)
    return(img_um)

def get_low_res_depth_grid(img):
    C0, S2, S1, N1, N2, I1, I2, T1, T2 = get_low_res_grid()
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

def get_low_res_depth_grid_values(img):
    C0, S2, S1, N1, N2, I1, I2, T1, T2 = get_low_res_grid()
    C0_value = np.mean(img[C0])
    S1_value = np.mean(img[S1])
    S2_value = np.mean(img[S2])
    I1_value = np.mean(img[I1])
    I2_value = np.mean(img[I2])
    T1_value = np.mean(img[T1])
    T2_value = np.mean(img[T2])
    N1_value = np.mean(img[N1])
    N2_value = np.mean(img[N2])
    return(C0_value, S1_value, S2_value,N1_value, N2_value, I1_value, I2_value, T1_value, T2_value)

def get_low_res_depth_grid_values_list(img, list_):
    C0, S2, S1, N1, N2, I1, I2, T1, T2 = get_low_res_grid()
    C0_value = np.mean(img[C0])
    S1_value = np.mean(img[S1])
    S2_value = np.mean(img[S2])
    I1_value = np.mean(img[I1])
    I2_value = np.mean(img[I2])
    T1_value = np.mean(img[T1])
    T2_value = np.mean(img[T2])
    N1_value = np.mean(img[N1])
    N2_value = np.mean(img[N2])
    list_.append(C0_value)
    return(list_)

def lower_bound_normal_eye_radialscan_oct3(img):
    '''Normal Macular Thickness Measurements in Healthy Eyes Using
           Stratus Optical Coherence Tomography'''
    C0_low = 192
    S1_low = 238
    S2_low = 223
    I1_low = 245
    I2_low = 197
    T1_low = 238
    T2_low = 196
    N1_low = 251
    N2_low = 232
    return(C0_low, S2_low, S1_low, N1_low, N2_low, I1_low, I2_low, T1_low, T2_low)

def upper_bound_normal_eye_radialscan_oct3():
    '''Normal Macular Thickness Measurements in Healthy Eyes Using
        Stratus Optical Coherence Tomography'''
    C0_upper = 232
    S1_upper = 272
    S2_upper = 255
    I1_upper = 275
    I2_upper = 223
    T1_upper = 264
    T2_upper = 224
    N1_upper = 283
    N2_upper = 260
    return(C0_upper, S2_upper, S1_upper, N1_upper, N2_upper, I1_upper, I2_upper, T1_upper, T2_upper)

def lower_bound_normal_eye_sd_oct():
    '''Macular Thickness by Age and Gender in Healthy Eyes Using Spectral Domain Optical Coherence Tomography'''
    C0_low = 209
    S1_low = 272
    S2_low = 234
    I1_low = 272
    I2_low = 230
    T1_low = 252
    T2_low = 216
    N1_low = 275
    N2_low = 253
    return(C0_low, S2_low, S1_low, N1_low, N2_low, I1_low, I2_low, T1_low, T2_low)

def upper_bound_normal_eye_sd_oct():
    '''Macular Thickness by Age and Gender in Healthy Eyes Using Spectral Domain Optical Coherence Tomography'''
    C0_upper = 249
    S1_upper = 308
    S2_upper = 260
    I1_upper = 303
    I2_upper = 256
    T1_upper = 298
    T2_upper = 248
    N1_upper = 310
    N2_upper = 284
    return(C0_upper, S2_upper, S1_upper, N1_upper, N2_upper, I1_upper, I2_upper, T1_upper, T2_upper)

def lower_bound_normal_eye_spectralis_oct():
    '''Macular Thickness by Age and Gender in Healthy Eyes Using Spectral Domain Optical Coherence Tomography'''
    C0_low = 250
    S1_low = 0
    S2_low = 0
    I1_low = 0
    I2_low = 0
    T1_low = 0
    T2_low = 0
    N1_low = 0
    N2_low = 0
    return(C0_low, S2_low, S1_low, N1_low, N2_low, I1_low, I2_low, T1_low, T2_low)

def upper_bound_normal_eye_spectralis_oct():
    '''Macular Thickness by Age and Gender in Healthy Eyes Using Spectral Domain Optical Coherence Tomography'''
    C0_upper = 291
    S1_upper = 10000
    S2_upper = 10000
    I1_upper = 10000
    I2_upper = 10000
    T1_upper = 10000
    T2_upper = 10000
    N1_upper = 10000
    N2_upper = 10000
    return(C0_upper, S2_upper, S1_upper, N1_upper, N2_upper, I1_upper, I2_upper, T1_upper, T2_upper)
