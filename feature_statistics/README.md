# feature statistics project

This repository implements the feature and time series calculations used in the project. 

"Changes in OCT Biomarkers of treatment naive patients in neovascular AMD using novel segmentation algorithm "

It consists of 3 python scripts, detailed below, for calculating feature statistics and generating time series data. As
well as jupyter notebooks for feature statistics calculations and visualuzation.

## segmentation statistics

To calculate the segmentation statistics for the segmented OCTs provide the path to the directory where the
segmentations are saved as SEG_DIR in the config.py file. Also specify the DATA_DIR in the config which will be 
where the segmentation_statistics.csv file is saved.

Then run the calculate_segmentation_statistics.py file. Important is that this file utilizes multiple cores, in 
order to calculate the statistics for all OCT segmentations utilitzing > 40 CPU cores is recommened. (last checked this
yielded a runtime of about 5 hours).

The calculate_segmentation_statistics.py file also has a commented for loop implementation of the statistics
calculation to be used for debugging.

##  all time series from the LODE project

In order to generate all time series available in the LODE project run the generate_full_sequences.py file. The file
requires the path to where you have saved your datasets, see overview presentation: 
 
 https://docs.google.com/presentation/d/1w6_uhX_lgCfP1dshMOtoNXXKuWRKu7rwEs5-HpgG-zI/edit#slide=id.ge2e0968ace_0_17
 
 When working on the ICB servers you can provide the following path: 
 
 **/storage/groups/ml01/datasets/projects/20181610_eyeclinic_niklas.koehler/joint_export**
 
 as the work_space directory. This has all the necessary file for running the script.
 
 The output of the program is the sequences.csv file which is saved in the workspace under "./sequence_data".
 
 ## structure time series
 
 The structured time series provide the longitudinal_properties.csv and longitudinal_properties_naive.csv files.
 This csv files contain time series data for sequences with a 1, 3 and 12 month visit and performs interpolation
 and carry over when exact values are specified, as detailed here: 
 
 https://docs.google.com/document/d/1bnKlKsCS5NdMMDu6XD5FqIdLow3sQ_64HwDj7UaPRns/edit?usp=sharing
 
 The csv files will be saved in the same directory as sequences.csv and need the same WORK_SPACE directory as the 
 generate_full_sequecnes.csv file.