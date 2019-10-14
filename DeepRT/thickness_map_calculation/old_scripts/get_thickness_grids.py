import os
import xml.etree.ElementTree as ET
import sys

sys.path.insert(0, '/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/')

from xml_data_class import *

path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/data_rep/first_data_export_xmls"

pat_paths = [os.path.join(path, i) for i in os.listdir(path)]

xml_paths = []
for pp in pat_paths:
    dates = os.listdir(pp)
    for d in dates:
        xml_paths.append(os.path.join(pp, d, "anom_explore_corrected_url.xml"))


def get_series(self):
    series = []
    study = self.get_study()
    for child in study:
        if 'Series' in str(child):
            series.append(child)
    return (series)


thickness_values = [[],[],[],[],[],[],[],[],[]]
patient_logging = [[],[],[]]
for xml in xml_paths:
    try:
        tree = ET.parse(xml)
        root = tree.getroot()
        data = xml_data(root, tree)
        study = data.get_study()

        #patient logging info
        pat_id = xml.split("/")[-3]
        study_date = data.get_study_date()

        all_series = study.findall("Series")
        for ser in all_series:
            laterality = ser.findall("./Laterality")[0].text

            zones = ser.findall("./ThicknessGrid/Zone")
            if zones != []:
                # add to logging
                patient_logging[0].append(pat_id)
                patient_logging[1].append(study_date)
                patient_logging[2].append(laterality)

                for k, zone in enumerate(zones):
                    thickness_values[k].append(zone.find("AvgThickness").text)
    except:
        print("path not working is: {}".format(xml))

thickness_pd = pd.DataFrame(thickness_values).T
thickness_pd = thickness_pd.rename(columns={0:"CO",1:"N1",2:"N2",3:"S1",4:"S2",5:"T1",6:"T2",7:"I1" ,8:"I2"})
patient_log_pd = pd.DataFrame(patient_logging).T
patient_log_pd = patient_log_pd.rename(columns={0:"patient_id",1:"study_date",2:"laterality"})

etdrs_data_pd = pd.concat([thickness_pd,patient_log_pd],axis=1)
etdrs_data_pd.to_csv("etdrs.csv")


