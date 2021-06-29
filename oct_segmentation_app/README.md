## This repository contains an example notebook to segment and visualize OCT images. 

Follow below steps to set up the segmentation.

1. First set up an anaconda enviroment from the *.yml file by running **conda env create -f conda_environment.yml**
2. Activate environment
3. run conda install -c anaconda ipykernel && python -m ipykernel install --user --name=oct_app

Then run the segmentation_demo.ipynb notebook with the set anaconda environment as the kernel.

### Note on evailable segmentation models

Currrently there are 5 models available to be used in an segmentation ensemble. This number could be increased to imporove performance further. In the notebook, set the "number of models" parameter to 5 in order to use all the available models for segmentating the OCT images.


