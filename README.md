# LODE
repository for all LODE projects

Steps to segment LMU data and generate AMD time series. 

#### 1. Given a trained segmentation model/ ensemble:

execute **feature_segmentation.evaluate.segment_dicom_ensemble.py** with specified ensebmle name, models and correctly configured WORK_SPACE, VOL_SAVE_PATH, EMBEDD_SAVE_PATH in the **feature_segmentation.segmentation_config.py** file.

The file_name variable need to be set to paths{1 | 2 | 3| 4}. This is the file name of the file containing the paths to the dicom files from the LMU eye clinic. They are split into four files so one can easily segment in paralell on 4 different gpus. 

Also, specify root_dir in the **feature_segmentation.evaluate.segment_dicom_ensemble.py** for adding sub directories to python path.

Output: **segmentation_statistics.csv** in WORK_SPACE/segmentation/feature_tables
#### 2. 
'''


'''
