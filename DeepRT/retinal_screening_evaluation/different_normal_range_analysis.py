import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_function import *

def in_normal_range(img,evaluating):
    if evaluating == 'radialscan_oct3':
        C0_upper, S2_upper, S1_upper, N1_upper, N2_upper, I1_upper, I2_upper,\
        T1_upper, T2_upper = upper_bound_normal_eye_radialscan_oct3()
        C0_low, S2_low, S1_low, N1_low, N2_low, I1_low, I2_low,\
        T1_low, T2_low = lower_bound_normal_eye_radialscan_oct3()

    if evaluating == 'sd_oct':
        C0_upper, S2_upper, S1_upper, N1_upper, N2_upper, I1_upper, I2_upper,\
        T1_upper, T2_upper = upper_bound_normal_eye_sd_oct()
        C0_low, S2_low, S1_low, N1_low, N2_low, I1_low, I2_low,\
        T1_low, T2_low = lower_bound_normal_eye_sd_oct()

    if evaluating == 'spectralis':
        C0_upper, S2_upper, S1_upper, N1_upper, N2_upper, I1_upper, I2_upper,\
        T1_upper, T2_upper = upper_bound_normal_eye_spectralis_oct()
        C0_low, S2_low, S1_low, N1_low, N2_low, I1_low, I2_low,\
        T1_low, T2_low = lower_bound_normal_eye_spectralis_oct()

    C0_value, S1_value, S2_value, N1_value, N2_value, I1_value, I2_value\
        ,T1_value, T2_value = get_low_res_depth_grid_values(img)

    C0_norm = C0_low<=C0_value<=C0_upper
    S1_norm = S1_low<=S1_value<=S1_upper
    S2_norm = S2_low<=S2_value<=S2_upper
    N1_norm = N1_low<=N1_value<=N1_upper
    N2_norm = N2_low<=N2_value<=N2_upper
    I1_norm = I1_low<=I1_value<=I1_upper
    I2_norm = I2_low<=I2_value<=I2_upper
    T1_norm = T1_low<=T1_value<=T1_upper
    T2_norm = T2_low<=T2_value<=T2_upper

    return(C0_norm, S2_norm, S1_norm, N1_norm, N2_norm, I1_norm, I2_norm, T1_norm, T2_norm)

def get_normal_range_dict(depth_map_path,evaluating):
    ''''''
    test_labels = os.listdir(depth_map_path)
    scale_factor = (1 / 1000.)

    normal_range_dict = {}
    for test_path in test_labels:
        record_name = get_record_name(test_path)
        test_label_dm = np.load(os.path.join(depth_map_path, test_path))
        if if_prediction(depth_map_path):
            test_label_dm = resize_prediction(test_label_dm)
        # test_label_dm_rescaled = rescale_oct_height(test_label_dm)
        low_res_label = get_low_res_depth_grid(test_label_dm)
        low_res_label_scaled = scale_zero_one(low_res_label, scale_factor)
        low_res_label_um = pixel_to_mu_meter(test_label_dm)
        normal_range_dict[record_name] = in_normal_range(low_res_label_um,evaluating)
    return(normal_range_dict)

def get_normal_range_statistics_pd(depth_map_path, evaluating):
    normal_range_dict = get_normal_range_dict(depth_map_path,evaluating)

    columns = {0: "C0_norm", 1: "S2_norm", 2: "S1_norm", 3: "N1_norm", 4: "N2_norm", 5: "I1_norm", 6: "I2_norm",
               7: "T1_norm", 8: "T2_norm"}
    normal_range_statistics = pd.DataFrame.from_dict(normal_range_dict, orient='index')
    normal_range_statistics = normal_range_statistics.rename(index=str, columns=columns)
    return(normal_range_statistics)






test_label_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/test_labels"
test_prediction_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/test_predictions"
save_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/normal_range_stats"

'''
evaluating = "radialscan_oct3"
paper_name = "Normal Macular Thickness Measurements in Healthy Eyes Using Stratus Optical Coherence Tomography"
normal_range_statistics_label = get_normal_range_statistics_pd(test_label_path)
normal_range_statistics_prediction = get_normal_range_statistics_pd(test_prediction_path)

normal_range_statistics_label.to_csv("./normal_range_stats/"+"_label_"+paper_name)
normal_range_statistics_prediction.to_csv("./normal_range_stats/"+"_prediction_"+paper_name)
'''
'''
evaluating = "sd_oct"
paper_name_sd_oct = "Macular Thickness by Age and Gender in Healthy Eyes Using Spectral Domain Optical Coherence Tomography"
normal_range_statistics_label = get_normal_range_statistics_pd(test_label_path,evaluating)
normal_range_statistics_prediction = get_normal_range_statistics_pd(test_prediction_path,evaluating)

normal_range_statistics_label.to_csv("./normal_range_stats/"+"_label_"+paper_name_sd_oct.replace(" ","_")+".csv")
normal_range_statistics_prediction.to_csv("./normal_range_stats/"+"_prediction_"+paper_name_sd_oct.replace(" ","_")+".csv")
'''

evaluating = "spectralis"
paper_name_sd_oct = "Comparison of Retinal Thickness in Normal Eyes Using Stratus and Spectralis Optical Coherence Tomography"
normal_range_statistics_label = get_normal_range_statistics_pd(test_label_path,evaluating)
normal_range_statistics_prediction = get_normal_range_statistics_pd(test_prediction_path,evaluating)

normal_range_statistics_label.to_csv("./normal_range_stats/"+"_label_"+paper_name_sd_oct.replace(" ","_")+".csv")
normal_range_statistics_prediction.to_csv("./normal_range_stats/"+"_prediction_"+paper_name_sd_oct.replace(" ","_")+".csv")

