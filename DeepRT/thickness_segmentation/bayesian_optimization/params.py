params = {}
params["img_shape"] = (256, 256, 3)
params["batch_size"] = 1
params["epochs"] = 30
params["learning_rate"] = 0.001
params["model"] = "bunet"
params["continuing_training"] = False
params["data_dir"] = "/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/data/clinic_data"
params["save_path"] = "./output_shuffled/"
params["loss_function"] = "dice_loss"
params["number_of_classes"] = 1
params["result_path"] = "./output"
params["save_predictions_path"] = "./results/best_patient_shuffled_model/predictions"

#### generator params
gen_params = {'dim': (params["img_shape"][0],params["img_shape"][1]),
          'batch_size': params["batch_size"],
          'n_channels': 3,
          'shuffle': True,
          'image_path': params["data_dir"],
          'label_path': params["data_dir"],
          "brightness_factor" : 0.5,
          "contrast_factor" : 0.7}