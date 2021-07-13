import glob
import pandas as pd
import numpy as np

not_anonymized_path = "./ForumBulkExport"
anonymized_path = "./ANONYMIZED_DICOMS"

anonymized_dicom = glob.glob(anonymized_path + "/*/*/*/*/*.dcm")
not_anonymized_dicom = glob.glob(not_anonymized_path + "/*/*.dcm")

anonymized_pd = pd.DataFrame(anonymized_dicom)
not_anonymized_pd = pd.DataFrame(not_anonymized_dicom)

anonymized_pd["dicom_name"] = anonymized_pd[0].str.split("/", expand=True)[6]
not_anonymized_pd["dicom_name"] = not_anonymized_pd[0].str.split("/", expand=True)[3]

# not_anonymized_records = [ar for ar in anonymized_pd["dicom_name"].tolist() if ar not in not_anonymized_pd[
# "dicom_name"].tolist()]

not_anonymized_dicom_names = np.setdiff1d(not_anonymized_pd["dicom_name"].tolist(), anonymized_pd["dicom_name"].tolist())

not_anonymized_dicom_names_pd = pd.DataFrame(not_anonymized_dicom_names, columns=["dicom_name"])

to_bo_anonymized_pd = pd.merge(not_anonymized_dicom_names_pd, not_anonymized_pd, left_on="dicom_name", right_on="dicom_name", how="left")

to_bo_anonymized_pd[0].to_csv("to_be_anonymized_CHECK.csv")