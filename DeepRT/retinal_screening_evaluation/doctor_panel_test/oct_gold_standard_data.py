import random
from jupyter_helper_functions import *

eval_path = "/home/olle/PycharmProjects/thickness_map_prediction/thickness_map_data/test_predictions"
data_path = "/home/olle/PycharmProjects/thickness_map_prediction/thickness_map_data"
gold_standard_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/doctor_panel_test/gold_standard_answers"

label_path = os.path.join(data_path, "thickness_maps")
prediction_path = eval_path
image_gen_path = os.path.join(data_path, "fundus")
gold_standard_mapping_path = os.path.join(gold_standard_path, "joint_gold_standard.csv")

gold_standard = pd.read_csv(gold_standard_mapping_path)
records = pd.read_csv(gold_standard_mapping_path).record_name.str.replace(".png","")
laterality_list = pd.read_csv(gold_standard_mapping_path).laterality
mapping = [[], []]

number_records = len(records)
idx_1 = random.sample(range(0, len(records)), number_records)

mapping[0] = records.str.replace(".png","").values
mapping[1] = idx_1
mapping_pd = pd.DataFrame.from_records(mapping).T.rename(columns={0: 'record_name', 1: 'pred_pseudo'})
mapping_pd.to_csv("./oct_gold_standard_test/test_3/mapping_fundus_label.csv")

# create data samples for doctors
label_paths = [os.path.join(label_path, i + ".npy") for i in records.values]
prediction_paths = [os.path.join(prediction_path, i + ".npy") for i in records.values]

save_path = "./oct_gold_standard_test/test_3/fundus_prediction_label"

for i, lp in enumerate(label_paths[0:number_records]):
    # try:
    name = lp.split("/")[-1].replace(".npy", "")
    laterality = laterality_list[i]
    save_name_pred = mapping_pd[mapping_pd["record_name"].values == name]["pred_pseudo"].values[0]

    #

    label_path_full = os.path.join(label_path, name + ".npy")
    pred_path_full = os.path.join(prediction_path, name + ".npy")
    image_path = os.path.join(image_gen_path, name + ".png")

    #plot_fundus_label_or_prediction_heidelberg_cs(pred_path_full, image_path, save_path,
    #                                              save_name_pred, laterality)

    plot_fundus_label_and_prediction(label_path_full, pred_path_full, image_path,
                                  save_path, name, laterality, gold_standard)

    #plot_fundus(label_path_full,image_path, save_name_pred, save_path,laterality)

# except:
#    print("Record not working is: {}".format(name))
