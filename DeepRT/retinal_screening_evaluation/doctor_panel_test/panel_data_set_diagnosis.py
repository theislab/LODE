import random
from jupyter_helper_functions import *

data_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/thickness_map_data_full_export"
abt_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/\
doctor_panel_test/oct_gold_standard_test/test_LMU_eyeclinic/full_abt.csv"
label_path = os.path.join(data_path,"thickness_maps")
prediction_path = os.path.join(data_path,"test_predictions")
image_path = os.path.join(data_path,"fundus")


mapping = [[],[],[]]
records = [i.replace(".npy","") for i in os.listdir(prediction_path)]
number_records = len(records)
idx_1 = random.sample(range(0, len(records)), number_records)
idx_2 = random.sample(range(len(records)+1, len(records)*2+1), number_records)

mapping[0] = records
mapping[1] = idx_1
mapping[2] = idx_2

mapping_pd = pd.DataFrame.from_records(mapping).T.rename(columns={0: 'record_name',
                                                                  1: 'label_pseudo', 2: 'pred_pseudo'})

#mapping_pd.to_csv("mapping_test_diagnosis_2.csv.csv")

full_abt = pd.read_csv(abt_path)
# create data samples for doctors
label_paths = [os.path.join(label_path,i) for i in os.listdir(prediction_path)]
prediction_paths = [os.path.join(prediction_path,i) for i in os.listdir(prediction_path)]

save_path = os.path.join("oct_gold_standard_test/test_LMU_eyeclinic",
                         "herorlds_fundus_prediction_label_exmaples")

for lp in label_paths[0:number_records]:
    try:
        name = lp.split("/")[-1].replace(".npy", "")
        laterality = name.split("_")[1]

        referral_answer_f = full_abt[full_abt.record_name==name]["r/nr_herold_f"].iloc[0]
        a_e_n_answer_f = full_abt[full_abt.record_name == name]["e/a/n_herold_f"].iloc[0]
        referral_answer_fp = full_abt[full_abt.record_name==name]["r/nr_herold_fp"].iloc[0]
        a_e_n_answer_fp = full_abt[full_abt.record_name == name]["e/a/n_herold_fp"].iloc[0]
        gold_standard_referral = full_abt[full_abt.record_name==name]["gold_standard_R/NR"].iloc[0]
        gold_standard_a_e_n = full_abt[full_abt.record_name == name]["gold_standard_A/N/E"].iloc[0]

        answers_dict = {"referral_answer_f":referral_answer_f,"a_e_n_answer_f":a_e_n_answer_f,
                        "referral_answer_fp":referral_answer_fp,"a_e_n_answer_fp":a_e_n_answer_fp,
                        "gold_standard_referral":gold_standard_referral,"gold_standard_a_e_n":gold_standard_a_e_n}

        plot_fundus_label_and_prediction(label_path, prediction_path, image_path,
                                         save_path, name, laterality, full_abt,answers = answers_dict)
    except:
        print("record not working: {}".format(lp))
