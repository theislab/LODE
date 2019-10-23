import glob
from multiprocess import Pool
from pydicom import read_file, write_file
import tqdm
import os
import numpy as np
import pandas as pd

ERASE_DIR = os.path.join('..', 'no_export')
EXPORT_DIR = os.path.join('..', 'fourth_export')
EXCEPTION_DIR = os.path.join('..', 'exception')
files = glob.glob(os.path.join('..', 'third_export', '**', '*.dcm'), recursive=True)

def person_names_callback(dataset, data_element):
    if data_element.VR == "PN":
        data_element.value = ""

def get_id(data):
    if data.PatientID[0:3] == "ps:":
        return data.PatientID[3:]
    else:
        return 'Unkown'


def handle_and_mv_export_dir(data, file):
    data.walk(person_names_callback)
    try:
        storage = data.MIMETypeOfEncapsulatedDocument
        if storage == 'application/pdf':
            os.remove(file)
        else:
            id = get_id(data)
            if not os.path.isdir(os.path.join(EXPORT_DIR, id)):
                os.mkdir(os.path.join(EXPORT_DIR, id))
            write_file(os.path.join(EXPORT_DIR, id, os.path.basename(file)), data)
            os.remove(file)
    except Exception:
        id = get_id(data)
        if not os.path.isdir(os.path.join(EXPORT_DIR, id)):
            os.mkdir(os.path.join(EXPORT_DIR, id))
        write_file(os.path.join(EXPORT_DIR, id, os.path.basename(file)), data)
        os.remove(file)

def mv_erase_dir(file):
    os.rename(file, os.path.join(ERASE_DIR, os.path.basename(file)))

def handle_file(file):
    try:
        data = read_file(file)
        modality = data.Modality
        if modality in ['OP', 'OPT', 'Opt', 'Op']:
            handle_and_mv_export_dir(data, file)
        else:
            mv_erase_dir(file)
    except Exception:
        try:
            os.rename(file, os.path.join(EXCEPTION_DIR, os.path.basename(file)))
        except Exception:
            pass

if __name__ == "__main__":
    pool = Pool(30)
    for _ in tqdm.tqdm(pool.imap_unordered(handle_file, files), total=len(files)):
        pass

