import random
from jupyter_helper_functions import *

eval_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/predictions"
data_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/\
data/stratified_and_patient_split/"

label_path = os.path.join(data_path,"test_labels")
prediction_path = os.path.join(eval_path,"test_predictions_total_var_filtered_clean_split")
image_gen_path = os.path.join(data_path,"test_images")

gold_standard_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/\
doctor_panel_test/oct_gold_standard_test"

xls = pd.ExcelFile(os.path.join(gold_standard_path,'test_2/GoldStandardsecondrun_pseudo.xlsx'))
orig = pd.read_excel(xls, 'Tabelle1')

miss_classified_records_path = os.path.join(gold_standard_path,"test_2/miss_classified_records.csv")
miss_classified_records = pd.read_csv(miss_classified_records_path)
records = miss_classified_records.record_name
# create data samples for doctors
label_paths = [os.path.join(label_path,i+".npy") for i in records.values]
prediction_paths = [os.path.join(prediction_path,i+".npy") for i in records.values]

save_path = "./oct_gold_standard_test/test_2/fundus_label_prediction"

def get_asnwer_and_goldstandard(name, miss_classified_records):
    answer = miss_classified_records[miss_classified_records.record_name == name].answer.values[0]
    gold_standard = miss_classified_records[miss_classified_records.record_name == name].gold_standard.values[0]
    return(answer, gold_standard)

for i, lp in enumerate(label_paths):
    #try:
    name = lp.split("/")[-1].replace(".npy", "")
    laterality = orig[orig.record_name == name].laterality.values[0]
    save_name_pred = miss_classified_records[miss_classified_records.record_name == name].id.values[0]

    label_path_full = os.path.join(label_path,name+".npy")
    pred_path_full = os.path.join(prediction_path, name + ".npy")
    image_path = os.path.join(image_gen_path, name + ".png")

    #get outcome
    answer, gold_standard = get_asnwer_and_goldstandard(name, miss_classified_records)

    plot_fundus_label_and_prediction(label_path_full, pred_path_full,image_path, save_path,
                                 save_name_pred, laterality, answer, gold_standard)
    #plot_fundus(label_path_full,image_path, save_name_pred, save_path,laterality)

#except:
    #    print("Record not working is: {}".format(name))

