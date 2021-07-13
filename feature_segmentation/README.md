# feature segmentation

The LODE/feature_segmentation repository implement the U-net architecture described here: https://docs.google.com/document/d/1bnKlKsCS5NdMMDu6XD5FqIdLow3sQ_64HwDj7UaPRns/edit?usp=sharing

## environment

to set up environment install the environment.yml file with anaconda.

## Usage

main necessary configuration required is in config.py where the 

- WORK_SPACE -  directory where you save data all files
- TRAIN_DATA_PATH - extension from WORK_SPACE where the images / masks are saved.
- DATA_SPLIT_PATH - if using a preset train, validation and split path

need to be set.

## Default set up
