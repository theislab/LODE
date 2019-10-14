import os
import xml.etree.ElementTree as ET
import sys
import pandas as pd
sys.path.insert(0, '/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/')

from xml_data_class import *

path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/first_data_export"

pat_paths = [os.path.join(path,i) for i in os.listdir(path)]

xml_paths = []
for pp in pat_paths:
    dates = os.listdir(pp)
    for d in dates:
        xml_paths.append(os.path.join(pp,d,"anom_explore_corrected_url.xml"))

def get_series(self):
    series = []
    study = self.get_study()
    for child in study:
        if 'Series' in str(child):
            series.append(child)
    return (series)

optical_nerv_id = [[],[]]
for xml in xml_paths:
    try:
        tree = ET.parse(xml)
        root = tree.getroot()
        data = xml_data(root, tree)
        study = data.get_study()

        all_series = study.findall("Series")

        for ser in all_series:
            exam = ser.findall("ExaminedStructure")
            series_id = ser.find("ReferenceSeries/ID")
            if ((series_id != [])==True):
                #get date infomration
                image = ser.findall("Image")[0]
                image_type = image.findall("ImageType/Type")[0].text
                if image_type == "LOCALIZER":
                    hour = ser.findall("Image")[0].findall("AcquisitionTime/Time/Hour")[0].text
                    min = ser.findall("Image")[0].findall("AcquisitionTime/Time/Minute")[0].text
                    sec = str(int(float(ser.findall("Image")[0].findall("AcquisitionTime/Time/Second")[0].text)))
                    time_stamp = "{:02d}".format(int(hour)) + "-" + "{:02d}".format(int(min))+ "-" + "{:02d}".format(int(sec))

                pat_id = xml.split("/")[-3]
                series_id = ser.find("ReferenceSeries/ID").text
                laterality = ser.findall("Laterality")[0].text
                study_date = data.get_study_date()
                id_string = pat_id +"_" + study_date +"_"+ laterality
                optical_nerv_id[0].append(id_string)
                optical_nerv_id[1].append(series_id)

    except:
        print("path not working is: {}".format(xml))

pd.DataFrame(optical_nerv_id).T.to_csv("series_ids_first_export.csv")


