#!/usr/bin/env python
"""
This library is able to read/write RAW OCT files expoted in the format -vol by Spectralis software.
"""

from struct import unpack
import numpy as np


def getOCTHdr(file):
    """
	Read the header of the .vol file and return it as a Python dictionary.
	"""

    # Read binary file
    with open(file, mode = 'rb') as file:
        fileContent = file.read()

    # Read raw hdr
    Version, SizeX, NumBScans, SizeZ, ScaleX, Distance, ScaleZ, SizeXSlo, SizeYSlo, ScaleXSlo, ScaleYSlo, FieldSizeSlo, ScanFocus, ScanPosition, ExamTime, ScanPattern, BScanHdrSize, ID, ReferenceID, PID, PatientID, Padding, DOB, VID, VisitID, VisitDate, GridType, GridOffset, GridType1, GridOffset1, ProgID, Spare = unpack(
        "=12siiidddiiddid4sQii16s16si21s3sdi24sdiiii34s1790s", fileContent[:2048])

    # Format hdr properly
    hdr = {'Version': Version.rstrip(), 'SizeX': SizeX, 'NumBScans': NumBScans, 'SizeZ': SizeZ, 'ScaleX': ScaleX,
           'Distance': Distance, 'ScaleZ': ScaleZ, 'SizeXSlo': SizeXSlo, 'SizeYSlo': SizeYSlo, 'ScaleXSlo': ScaleXSlo,
           'ScaleYSlo': ScaleYSlo, 'FieldSizeSlo': FieldSizeSlo, 'ScanFocus': ScanFocus,
           'ScanPosition': ScanPosition.rstrip(), 'ExamTime': ExamTime,
           'ScanPattern': ScanPattern, 'BScanHdrSize': BScanHdrSize, 'ID': ID.rstrip(),
           'ReferenceID': ReferenceID.rstrip(), 'PID': PID, 'PatientID': PatientID.rstrip(), 'DOB': DOB, 'VID': VID,
           'VisitID': VisitID.rstrip(), 'VisitDate': VisitDate, 'GridType': GridType, 'GridOffset': GridOffset,
           'GridType1': GridType1, 'GridOffset1': GridOffset1, 'ProgID': ProgID.rstrip()}

    return hdr


def getSLOImage(file, hdr):
    """
	Read the SLO image stored in the  .vol file and returns it as a Numpy array.
	"""

    # Read binary file
    with open(file, mode = 'rb') as file:
        fileContent = file.read()

    # Read SLO image
    SizeXSlo = hdr['SizeXSlo']
    SizeYSlo = hdr['SizeYSlo']
    SloSize = SizeXSlo * SizeYSlo
    SloOffset = 2048
    SloImg = unpack(
        '=' + str(SloSize) + 'B', fileContent[SloOffset:(SloOffset + SloSize)])
    SloImg = np.asarray(SloImg, dtype = 'uint8')
    SloImg = SloImg.reshape(SizeXSlo, SizeYSlo)

    return SloImg

def getOCTBScan(file, hdr, n):
    """
	Read the OCT image n stored in the  .vol file and returns it as a Numpy array.
	"""

    # Read binary file
    with open(file, mode = 'rb') as file:
        fileContent = file.read()

    # Read SLO image
    SizeXSlo = hdr['SizeXSlo']
    SizeYSlo = hdr['SizeYSlo']
    SloSize = SizeXSlo * SizeYSlo
    SloOffset = 2048
    OCTOffset = SloOffset + SloSize

    BsBlkSize = hdr['BscanHdrSize'] + hdr['SizeX'] * hdr['SizeZ'] * 4

    OCTheader = unpack(
        '12', fileContent[OCTOffset:OCTOffset + n*hdr['SizeX']])


    OCTImg = unpack(
        '=' + str(BsBlkSize) + 'B', fileContent[OCTOffset + (n-1)*BsBlkSize:OCTOffset + n*BsBlkSize])
    OCTImg = np.asarray(OCTImg, dtype = 'uint8')
    OCTImg = OCTImg.reshape(hdr['SizeX'], hdr['SizeZ'])
    return OCTImg
