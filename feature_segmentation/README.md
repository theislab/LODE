# feature segmentation

The LODE/feature_segmentation repository implement the U-net architecture described here: https://docs.google.com/document/d/1bnKlKsCS5NdMMDu6XD5FqIdLow3sQ_64HwDj7UaPRns/edit?usp=sharing

To reproduce the results in the document please edit the params.json file with model and training config parameters as stated in the above manuscript and rerun.

## environment

to set up environment install the environment.yml file with anaconda.

## usage

main necessary configuration required is in config.py where the 

- WORK_SPACE -  directory where you save data all files
- TRAIN_DATA_PATH - where the images / masks are saved.
- DATA_SPLIT_PATH - path to, if using a preset train, validation and split path

need to be set.

## default set up

if working on the ICB / Helmholtz compute:

TRAIN_DATA_PATH = "/storage/groups/ml01/datasets/projects/20201010_LODE_segmentation/feature_segmentation/data/records"
DATA_SPLIT_PATH = "/storage/groups/ml01/datasets/projects/20201010_LODE_segmentation/feature_segmentation/data/data_split"

## general

in the /storage/groups/ml01/datasets/projects/20201010_LODE_segmentation/feature_segmentation/ directory one find the data files for the segmentation project.
This includes unrevised and revised json file. Please use revised (by default stored in the above written TRAIN_DATA_PATH). During the annotations process standards were changed, hence the unrevised annotations sometimes contains annotation errors. 

### generating segmentation masks from json annotations files

Please see /label_conversion/labelme2voc.py file. To use, edit the python file with the correct path to the directory holding the json file and then run the files.

The annotation masks as well as visualizations are saved in the same directory as the label conversion python file. 
