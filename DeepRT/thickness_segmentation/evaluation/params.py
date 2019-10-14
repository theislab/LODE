params = {}
params["img_shape"] = (256, 256, 3)
params["batch_size"] = 1
params["epochs"] = 1
params["learning_rate"] = 0.001
params["continuing_training"] = True
params["data_dir"] = "/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/data"
params["save_path"] = "./output_spectralis_topcon"
params["loss_function"] = "dice_loss"
params["number_of_classes"] = 1

#### generator params
gen_params = {'dim': (params["img_shape"][0],params["img_shape"][1]),
          'batch_size': params["batch_size"],
          'n_channels': 3,
          'shuffle': True,
          'image_path': params["data_dir"],
          'label_path': params["data_dir"],
          "brightness_factor" : 0.5,
          "contrast_factor" : 0.7}