import os
import glob
import pydicom as py
import cv2

main_path = "/storage/groups_new/ml01/datasets/raw/2018_LMUAugenklinik_niklas.koehler/Studies/Optical Coherence Tomography Scanner"
save_dir = "/storage/groups_new/ml01/datasets/projects/20181610_eyeclinic_niklas.koehler/oct_images"

dicom_paths = glob.glob(main_path + "/**/**/**/*.dcm")

for dc in dicom_paths:
    dicom_ = py.read_file(dc)
    study_date = dicom_.StudyDate
    series_number = dicom_.SeriesNumber
    patient_id = dicom_.PatientID.replace("ps:","")
    image_Laterality = dicom_.ImageLaterality

    main_save_name = str(patient_id)+"_"+str(image_Laterality)+"_"+str(study_date)+"_"+str(series_number)
    oct_stack = dicom_.pixel_array

    for iter_ in range(0,max(range(oct_stack.shape[0]))+1):
        oct_ = oct_stack[iter_,:,:]
        cv2.imwrite(os.path.join(save_dir,main_save_name+"_"+str(iter_)+".png"),oct_)
