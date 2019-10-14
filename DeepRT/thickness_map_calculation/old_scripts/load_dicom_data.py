import os
from pydicom import read_file
import pandas as pd
import time
import numpy as np
import logging
def filter_dicom(dc):

    manuf = dc.Manufacturer
    study_descr = dc.StudyDescription
    series_description = dc.SeriesDescription
    pixel_shape = dc.pixel_array.shape
    if (manuf == "Heidelberg Engineering") & (study_descr == 'Makula (OCT)') & (series_description == 'Volume IR')\
            & (pixel_shape[0] == 49):
        return(dc)
    else:
        logging.info("Dicom did not contain correct data, see values for mauf, study desc, series desc and pixel shape: "
              "{},{},{},{}".format(manuf,study_descr,series_description,pixel_shape))
        return(None)
def get_oct_data(dc,path):
    filtered_dicoms = []
    fundus_positions = []
    image_positions = []
    stack_positions = []
    x_scales = []
    y_scales = []
    x_starts = []
    y_starts = []
    x_ends = []
    y_ends = []
    for i in range(0,len(dc.PerFrameFunctionalGroupsSequence)):
        filtered_dicoms.append(path)
        y_start = dc.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[0].ReferenceCoordinates[0]
        x_start = dc.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[0].ReferenceCoordinates[1]
        y_end = dc.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[0].ReferenceCoordinates[2]
        x_end = dc.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[0].ReferenceCoordinates[3]
        image_position = dc.PerFrameFunctionalGroupsSequence[i].PlanePositionSequence[0].ImagePositionPatient
        stack_position = dc.PerFrameFunctionalGroupsSequence[i].FrameContentSequence[0].InStackPositionNumber
        y_scale = dc.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]
        x_scale = dc.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[1]

        image_positions.append(image_position)
        stack_positions.append(stack_position)
        x_scales.append(x_scale)
        y_scales.append(y_scale)
        y_starts.append(y_start)
        x_starts.append(x_start)
        y_ends.append(y_end)
        x_ends.append(x_end)

    return(filtered_dicoms, image_positions, stack_positions, x_scales,y_scales,y_starts,x_starts,y_ends,x_ends)


def get_patient_data(dicom_paths):

    patient_id = []
    image_types = []
    filtered_dicoms = []
    series_id = []
    study_date = []
    image_shape = []
    laterality = []
    series_numbers = []
    oct_list = [[],[],[],[],[],[],[],[],[]]
    #load all dicom files and append to dicom list
    for path in dicom_paths:
        dc = read_file(path)
        #try if dicom has all files
        dc = filter_dicom(dc)
        if dc is None:
            return(None)
        if dc is not None:
            filtered_dicoms.append(path)
            #remove all non digits from string
            patient_id.append(''.join(c for c in dc.PatientID if c.isdigit()))
            laterality.append(dc.ImageLaterality)
            image_types.append(dc.ImageType)
            series_numbers.append(dc.SeriesNumber)
            study_date.append(dc.StudyDate)
            image_shape.append(dc.pixel_array.shape)

            #get all oct data
            filtered_dicoms, image_positions, stack_positions, x_scales, y_scales, y_starts, x_starts\
                ,y_ends, x_ends = get_oct_data(dc,path)

            oct_list[0] += filtered_dicoms
            oct_list[1] += image_positions
            oct_list[2] += stack_positions
            oct_list[3] += x_scales
            oct_list[4] += y_scales
            oct_list[5] += y_starts
            oct_list[6] += x_starts
            oct_list[7] += y_ends
            oct_list[8] += x_ends

    oct_dict = {"dicom_paths": oct_list[0],"image_positions":oct_list[1],
                    "stack_positions":oct_list[2],"x_scales":oct_list[3],
                    "y_scales":oct_list[4], "y_starts":oct_list[5],"x_starts":oct_list[6],
                "y_ends":oct_list[7],"x_ends":oct_list[8],}

    patient_dict = {"dicom_paths": filtered_dicoms,"patient_id":patient_id, "series_numbers":series_numbers,
                    "image_types":image_types,"series_id":series_id,
                    "study_date":study_date, "image_shape":image_shape, "laterality":laterality}

    patient_pd = pd.DataFrame.from_dict(patient_dict, orient="index").T
    oct_pd = pd.DataFrame.from_dict(oct_dict, orient="index").T

    patient_full_pd = pd.concat((patient_pd,oct_pd), axis=1)
    return(patient_full_pd)

def get_dicom_look_up(dicom_path):
    patient_pd = get_patient_data([dicom_path])
    return(patient_pd)

def get_full_look_up():
    number_dicoms = []
    start = time.time()

    #initialize main data frame
    columns = {'series_id', 'dicom_paths', 'image_types',
               'laterality','image_shape', 'study_date',
               'series_numbers','patient_id','dicom_paths',
               'x_ends', 'x_starts', 'stack_positions', 'y_ends',
               'x_scales', 'y_starts', 'image_positions', 'y_scales'}
    main_frame = pd.DataFrame(columns=columns)
    patient_dir = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/full_data_export/"
    #get all patient paths
    patient_paths = [os.path.join(patient_dir, sub_dir) for sub_dir in os.listdir(patient_dir)]
    for patient_path in patient_paths:
        study_paths = [os.path.join(patient_path, sub_dir) for sub_dir in os.listdir(patient_path)]
        for study_path in study_paths:
            try:
                dicom_paths = [os.path.join(study_path, sub_dir) for sub_dir in os.listdir(study_path)]
                number_dicoms.append(len(dicom_paths))
                patient_pd = get_patient_data(dicom_paths)
                main_frame = pd.concat((patient_pd, main_frame))

                if np.sum(number_dicoms) % 1000 == 0:
                    print("just processed 1000 records, time elapsed")

                    print("just completed patient: {}".format(patient_path))

            except:
                print("paths not working are: {},{}".format(patient_path, study_path))
                continue

    main_frame.to_csv("./data_overview.csv")
    end = time.time()
    print(end - start)
