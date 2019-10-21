import pandas as pd
import os
import ezodf
from shutil import copyfile
from sklearn.metrics import confusion_matrix
from jupyter_helper_functions import *
def read_ods(filename, sheet_no=0, header=0):
    tab = ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({col[header].value:[x.value for x in col[header+1:]]
                         for col in tab.columns()})

eval_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/doctor_panel_test/test_2"

test_records_file = "records_test_2"
key_file = "mapping_test_2.csv"
answer_file = "doctors_fillout_form_test.ods"

key_pd = pd.read_csv(os.path.join(eval_path,key_file))
answer_pd = read_ods(os.path.join(eval_path,answer_file))

test_records = os.listdir(os.path.join(eval_path,test_records_file))


label_sheet = pd.merge(key_pd,answer_pd,left_on="label_pseudo",right_on='ids',how="left")

label_sheet = label_sheet[label_sheet['ids'].notnull()][['record_name','Pathological','ids']]

pred_sheet = pd.merge(key_pd[["record_name","pred_pseudo"]][key_pd["pred_pseudo"].notnull()],answer_pd,left_on="pred_pseudo",right_on='ids',how="left")
pred_sheet  = pred_sheet[pred_sheet['ids'].notnull()][['record_name','Pathological','ids']]

comparison_sheet = pd.merge(label_sheet, pred_sheet, left_on="record_name", right_on= "record_name", how="outer")

label_votes = comparison_sheet.Pathological_x.astype(bool)
pred_votes = comparison_sheet.Pathological_y.astype(bool)
save_viz = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/doctor_panel_test/test_2"
plot_confusion_matrix(label_votes, pred_votes, num_classes=2,save_path=save_viz)

truemaps_path_predmaps_nonpath = comparison_sheet[(comparison_sheet.Pathological_x == 1) & (comparison_sheet.Pathological_y == 0)]
ids_path_nonpath = list(truemaps_path_predmaps_nonpath.ids_x.astype(int)) + list(truemaps_path_predmaps_nonpath.ids_y.astype(int))

save_mistakes = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/doctor_panel_test/test_2/true-path_pred-nonpath_mistakes"
iter_ = 0

for i in truemaps_path_predmaps_nonpath.ids_x.astype(int):
    #get pred pseudo id belonging to the label pseudo id
    pred_psuedoid = truemaps_path_predmaps_nonpath[truemaps_path_predmaps_nonpath.ids_x == i].ids_y.values[0].astype(int)
    #get reocrd actual name for id
    record = truemaps_path_predmaps_nonpath[truemaps_path_predmaps_nonpath.ids_x == i].record_name.values[0]
    #get path for copying
    label_im_path = os.path.join(os.path.join(eval_path, test_records_file), str(i) + ".png")
    pred_im_path = os.path.join(os.path.join(eval_path, test_records_file), str(pred_psuedoid) + ".png")
    #copy files with true record name as id in naming
    copyfile(label_im_path,os.path.join(save_mistakes,record+"_true_thicknessmap_"+str(iter_)+"_"+str(i)+"_.png"))
    copyfile(pred_im_path, os.path.join(save_mistakes,record+"_predicted_thicknessmap_"+str(iter_)+"_"+str(pred_psuedoid)+"_.png"))
    iter_ += 1

iter_ = 0

truemaps_nonpath_predmaps_path = comparison_sheet[(comparison_sheet.Pathological_x == 0) & (comparison_sheet.Pathological_y == 1)]
save_mistakes = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/doctor_panel_test/test_2/true-nonpath_pred-path_mistakes"

for i in truemaps_nonpath_predmaps_path.ids_x.astype(int):
    #get pred pseudo id belonging to the label pseudo id
    pred_psuedoid = truemaps_nonpath_predmaps_path[truemaps_nonpath_predmaps_path.ids_x == i].ids_y.values[0].astype(int)
    #get reocrd actual name for id
    record = truemaps_nonpath_predmaps_path[truemaps_nonpath_predmaps_path.ids_x == i].record_name.values[0]
    #get path for copying
    label_im_path = os.path.join(os.path.join(eval_path, test_records_file), str(i) + ".png")
    pred_im_path = os.path.join(os.path.join(eval_path, test_records_file), str(pred_psuedoid) + ".png")
    #copy files with true record name as id in naming
    copyfile(label_im_path,os.path.join(save_mistakes,record+"_true_thicknessmap_"+str(iter_)+"_"+str(i)+"_.png"))
    copyfile(pred_im_path, os.path.join(save_mistakes,record+"_predicted_thicknessmap_"+str(iter_)+"_"+str(pred_psuedoid)+"_.png"))
    iter_ += 1

save_correct = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/doctor_panel_test/test_2/correct_records"

aligning_predmaps_path = comparison_sheet[((comparison_sheet.Pathological_x == 1) & (comparison_sheet.Pathological_y == 1))\
                                                  |(comparison_sheet.Pathological_x == 0) & (comparison_sheet.Pathological_y == 0)]

for i in aligning_predmaps_path.ids_x.astype(int):
    #get pred pseudo id belonging to the label pseudo id
    pred_psuedoid = aligning_predmaps_path[aligning_predmaps_path.ids_x == i].ids_y.values[0].astype(int)
    #get reocrd actual name for id
    record = aligning_predmaps_path[aligning_predmaps_path.ids_x == i].record_name.values[0]
    #get path for copying
    label_im_path = os.path.join(os.path.join(eval_path, test_records_file), str(i) + ".png")
    pred_im_path = os.path.join(os.path.join(eval_path, test_records_file), str(pred_psuedoid) + ".png")
    #copy files with true record name as id in naming
    copyfile(label_im_path,os.path.join(save_correct,record+"_true_thicknessmap_"+str(iter_)+"_"+str(i)+"_.png"))
    copyfile(pred_im_path, os.path.join(save_correct,record+"_predicted_thicknessmap_"+str(iter_)+"_"+str(pred_psuedoid)+"_.png"))
    iter_ += 1

print("hello")
