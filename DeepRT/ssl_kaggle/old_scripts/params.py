import sys

sys.dont_write_bytecode = True
# 32
# 28
params = {}
params["batch_size"] = 3
params["img_shape"] = (128, 128, 3)
params["epochs"] = 30
params["learning_rate"] = 0.001
#set to true if weights should not be randomly inititalized
params["continuing_training"] = True
params["save_path"] = "./ssl_tp"
#set what weights to init network with
params["weights"] = "thickness_map"
params["generator_type"] = "simple"
#set parameter for under or oversampling
params["sampling"] = "undersampling"
#set path to proportion of data set to be trained on
params["id_path"] = "/media/olle/Seagate/kaggle/id_files/unbalanced/hundred"

#### generator params
gen_params = {'dim': (params["img_shape"][0],params["img_shape"][1]),
          'batch_size': params["batch_size"],
          'n_channels': 3,
          'shuffle': True,
          'fundus_path': "/media/olle/Seagate/kaggle/train",
          'label_path': "/media/olle/Seagate/kaggle",
          "brightness_factor": 0.5,
          "contrast_factor": 0.7}

#'fundus_path':"/home/icb/olle.holmberg/projects/data/thickness_map_data/fundus",
#'thickness_path':"/home/icb/olle.holmberg/projects/data/thickness_map_data/thickness_maps"