import os
import numpy as np
from multiprocess import Pool
import tqdm
from pydicom import read_file

#helper functions
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

directory_list = os.listdir('../exports')
# get all patients from directories
patient_list = []
for directory in directory_list:
    patient_list.extend(os.listdir(os.path.join('..', 'exports', directory)))
patient_list = list(set(patient_list))

# get a sample of the patients
# patient_list = np.random.choice(patient_list, 100, replace=False)

def move_dcm_file(description, patient, date, file):
    try:
        destination = os.path.join('..', 'Studies', description, patient, date)
        if not os.path.isdir(destination):
            os.makedirs(destination)
        os.rename(file, os.path.join(destination, os.path.basename(file)))
    except Exception as ex:
        print("A: {} \n {}".format(file, ex))
        
def handle_patient(patient):
    file_list = []
    for directory in directory_list:
        try:
            file_list.extend(listdir_fullpath(os.path.join('..', 'exports', directory, patient)))    
        except FileNotFoundError:
            pass
        except Exception as ex:
            print("B: {}: \n {}".format(patient, ex))
            
    for file in file_list:
        try:
            data = read_file(file)
            # get description
            try:
                description = data.StudyDescription
                if description == '':
                    description = 'Empty Description'
                if description == 'Makula (OCT)':
                    try:
                        description = data.AcquisitionDeviceTypeCodeSequence[0].CodeMeaning
                    except Exception:
                        description = 'Other Makula (OCT) Related'
            except Exception:
                description = 'No Description'
            # get date
            try:
                date = data.StudyDate
                if date == '':
                    date = 'Empty Date'
            except Exception:
                date = 'No Date'
            move_dcm_file(description, patient, date, file)
        except Exception as ex:
            print("C: {}: \n {}".format(file, ex))

    for directory in directory_list:
        try:
            original_dir = os.path.join('..', 'exports', directory, patient)
            if len(os.listdir(original_dir)) == 0:
                os.rmdir(original_dir)
            else:
                print("D:{} is not Empty!".format(original_dir))
        except FileNotFoundError:
            pass
        except Exception as ex:
            print("E: {}".format(ex))
            
if __name__ == "__main__":
    pool = Pool(30)
    for _ in tqdm.tqdm(pool.imap_unordered(handle_patient, patient_list), total=len(patient_list)):
        pass
