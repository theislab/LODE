
import random
from jupyter_helper_functions import *
import gc
eval_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/predictions"
data_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/\
data/stratified_and_patient_split/"


record_info_path = ""

label_path = os.path.join(data_path,"export_2_filtered_total_variation/test_labels")
prediction_path = os.path.join(eval_path,"test_predictions_mse_reg")
image_path = os.path.join(data_path,"export_2_filtered_total_variation/test_images")
first_export_mapping_path = os.path.join(data_path,"export_2_filtered_total_variation/test_patient_mapping_first_export.csv")
second_export_mapping_path = os.path.join(data_path,"export_2_filtered_total_variation/test_patient_mapping_longitudinal_export.csv")

mapping = [[],[],[]]
records = [i.replace(".npy","") for i in os.listdir(prediction_path)]
number_records = len(records)
number_records = 100
idx_1 = random.sample(range(0, len(records)), number_records)
idx_2 = random.sample(range(len(records)+1, len(records)*2+1), number_records)

mapping[0] = records
mapping[1] = idx_1
mapping[2] = idx_2

mapping_pd = pd.DataFrame.from_records(mapping).T.rename(columns={0: 'record_name',
                                                                  1: 'label_pseudo', 2: 'pred_pseudo'})

mapping_pd.to_csv("mapping_test_4.csv")

first_export_mapping = pd.read_csv(first_export_mapping_path)
second_export_mapping = pd.read_csv(second_export_mapping_path)

#add record name to first csv
first_export_mapping["record_name"] = first_export_mapping.patient_id.astype(str) + "_" + \
                                      first_export_mapping.study_date.astype(str) +"_" + \
                                      first_export_mapping.laterality.astype(str)+ "_" + \
                                      first_export_mapping.series_time.astype(str)

# create data samples for doctors
label_paths = [os.path.join(label_path,i) for i in os.listdir(prediction_path)]
prediction_paths = [os.path.join(prediction_path,i) for i in os.listdir(prediction_path)]

save_path = "./records_test_4"

for lp in label_paths[0:number_records]:
    try:
        name = lp.split("/")[-1].replace(".npy", "").replace(".png","")
        try:
            laterality = second_export_mapping[second_export_mapping.record_name == name].laterality.values[0]
        except:
            pass
        try:
            laterality = first_export_mapping[first_export_mapping.record_name == name].laterality.values[0]
        except:
            pass
        save_name = mapping_pd[mapping_pd["record_name"].values == name]["label_pseudo"].values[0]
        plot_label_heatmap_pair(lp, save_path, save_name,laterality,prediction=False)
        save_name_pred = mapping_pd[mapping_pd["record_name"].values == name]["pred_pseudo"].values[0]
        plot_label_heatmap_pair(os.path.join(prediction_path,name+".npy"), save_path, save_name_pred,laterality, prediction=True)
    except:
        print("Record not working is: {}".format(name))

