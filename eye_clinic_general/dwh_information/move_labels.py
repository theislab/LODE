import pandas as pd
import os
from multiprocessing import Pool
from pydicom import read_file, write_file

pseudo_id = pd.read_csv('./pseudo_id_key.csv', delimiter=';').set_index('PATNR')

def anonymize(pat_nr):
    print(pat_nr)
    if pat_nr[0:3] == "ps:":
        return pat_nr
    if pat_nr == '':
        return ''
    try:
        match = pseudo_id.at[int(pat_nr), 'pseudo_id']
        return "ps:"+str(match)
    except Exception:
        return None

def pseudo_anonymize(dicom_file):
    try:
        data = read_file(dicom_file)
        if hasattr(data, 'PatientID'):
            data.PatientID = anonymize(data.PatientID)
        if hasattr(data, 'PatientName'):
            data.PatientName = ''
        if hasattr(data, 'PatientBirthDate'):
            data.PatientBirthDate = data.PatientBirthDate[0:4] + "0101"
        if hasattr(data, 'PerformingPhysicianName'):
            data.PerformingPhysicianName = ''
        write_file(dicom_file, data)
    except:
        pass 

def update_dir(dir):
    buffer = []
    for parent, _, file_list in os.walk(dir):
        for file in file_list:
            full_file = os.path.join(parent, file)
            pseudo_anonymize(full_file)
            print(full_file)


if __name__ == '__main__':
    data_path = os.path.join('..', 'data_export')
    dir_list = [os.path.join(data_path, dir) for dir in os.listdir(data_path)]
    pool = Pool(15)
    pool.map(update_dir, dir_list)
