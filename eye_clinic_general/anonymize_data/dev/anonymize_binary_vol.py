import os
import numpy as np
from eye_clinic_general.anonymize_data.binary_vol_utils import getOCTHdr, getSLOImage, getOCTBScan

bin_dir = "/home/olle/PycharmProjects/LODE/workspace/anonymize_examples/binary"

record_name = "Basani_B_849525.vol"

file_path = os.path.join(bin_dir, record_name)

hdr = getOCTHdr(file_path)

if 'OCT' in str(hdr['Version']):
    sol = getSLOImage(file_path, hdr)

    np.empty(shape=(hdr["NumBScans"], ))
    for i in range(1, hdr["NumBScans"] + 1):
        oct_b_scan = getOCTBScan(file_path, hdr, i)