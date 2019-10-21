import math
import pandas as pd
import numpy as np


class xml_data(object):
    def __init__(self, root, tree):
        self.root = root
        self.tree = tree

    # first class function return elements and ID's of tree
    def get_patient(self):
        '''get the Patient elem of tree'''
        BODY = self.root[0]
        for child in BODY:
            if 'Patient' in str(child):
                patient = child
        return (child)

    def get_study(self):
        patient = self.get_patient()
        for child in patient:
            if 'Study' in str(child):
                study = child
        return (study)

    def get_series(self):
        series = []
        study = self.get_study()
        for child in study:
            if 'Series' in str(child):
                series.append(child)
        return (series)

    def get_images(self):
        '''Get all image elements from xml'''
        Images = []
        series = self.get_series()
        for ser in series:
            for child in ser:
                if "Image" in child.tag and 'Num' not in child.tag:
                    Images.append(child)
        return (Images)

    def get_patient_id(self):
        patient = self.get_patient()
        for child in patient:
            if 'PatientID' in str(child):
                patientID = child
        return patientID

    def get_study_id(self):
        study = self.get_study()
        for child in study:
            if 'ID' in str(child) and 'Study' not in str(child):
                studyID = child
        return studyID

    def get_series_id(self):
        series = self.get_series()
        for child in series:
            if 'ID' in str(child) and 'Series' not in str(child) and 'Prog' not in str(child):
                seriesID = child
        return seriesID

    def get_parentmap(self):
        return (dict((c, p) for p in self.tree.getiterator() for c in p))

    def pID_orNAN(self):
        'This function returns the PatientID if exists otherwise Nan. Note that PatientID is the ID in DWH'
        patient = self.get_patient()
        l = []
        r = []
        for child in patient:
            if "PatientID" in str(child):
                l.append(child.text)
            if "PatientID" not in str(child):
                l.append("No PatientID")
            for i in l:
                if "No" not in i:
                    r.append(i)
        if not r:
            r.append("NaN")
        return (r[0])

    ##Second series of function returns tables of data from XML tree##
    def get_patient_birthdate(self):
        patient = self.get_patient()
        for child in patient:
            if 'Birthdate' in str(child):
                date = child[0]
                for children in date:
                    if 'Year' in str(children):
                        year = children.text
                    if 'Month' in str(children):
                        month = children.text
                    if 'Day' in str(children):
                        day = children.text

        return ('-'.join((year, month, day)))

    def get_study_date(self):
        '''return the date the study was made sa string'''
        study = self.get_study()
        for child in study:
            if 'StudyDate' in str(child):
                date = child[0]
                for children in date:
                    if 'Year' in str(children):
                        year = children.text
                    if 'Month' in str(children):
                        month = children.text
                    if 'Day' in str(children):
                        day = children.text

        return ('-'.join((year, month, day)))

    def get_image_aquisition_time(self):
        '''return the date the study was made sa string'''
        image = self.get_images()
        for child in image:
            for children in child:
                if 'AcquisitionTime' in str(children):
                    time = child[0]
                    for children in time:
                        if 'Hour' in str(children):
                            hour = children.text
                        if 'Minute' in str(children):
                            minute = children.text
                        if 'Second' in str(children):
                            second = children.text

        return ('-'.join((hour, minute, second)))

    def get_thickness_grid_elem(self):
        '''the function returns the element representing the thickness grid'''
        thickness_grids = []
        series = self.get_series()
        for ser in series:
            for child in ser:
                if 'ThicknessGrid' in str(child):
                    thickness_grids.append(child)
        return (thickness_grids)

    def get_thicknessgrid(self):
        '''This function returns a pandas table with the Thickness grid for each series'''

        Type = []
        CentralThickness = []
        MinCentralThickness = []
        MaxCentralThickness = []
        TotalVolume = []
        X_cetner_pos = []
        Y_cetner_pos = []
        Series_ID = []
        parent_map = self.get_parentmap()
        series = self.get_series()
        for ser in series:
            for child in ser:
                if 'ThicknessGrid' in str(child):
                    series_id = parent_map[child]
                    Series_ID.append(series_id.findall('ID')[0].text)
                    # get direct children
                    Type.append(child.findall('Type')[0].text)
                    CentralThickness.append(child.findall('CentralThickness')[0].text)
                    MinCentralThickness.append(child.findall('MinCentralThickness')[0].text)
                    MaxCentralThickness.append(child.findall('MaxCentralThickness')[0].text)
                    TotalVolume.append(child.findall('TotalVolume')[0].text)

                    # get center position
                    center_pos = child.findall('CenterPos')[0]
                    for children in center_pos:
                        X_cetner_pos.append(children[0].text)
                        Y_cetner_pos.append(children[1].text)

        thicknessgrip_d = {'Series_ID': Series_ID, 'Type': Type, 'CentralThickness': CentralThickness,
                           'MinCentralThickness': MinCentralThickness,
                           'MaxCentralThickness': MaxCentralThickness, 'TotalVolume': TotalVolume,
                           'X_cetner_pos': X_cetner_pos, 'Y_cetner_pos': Y_cetner_pos}

        return (pd.DataFrame(data=thicknessgrip_d))

    def get_thicknessgrid_zones(self):
        '''This function return a pandas with thickness information for each zone'''
        Name = []
        AvgThickness = []
        Volume = []
        Series_ID = []
        thick_grid = self.get_thickness_grid_elem()
        parent_map = self.get_parentmap()
        for child in thick_grid:
            for children in child:
                if 'Zone' in str(children):
                    series = parent_map[parent_map[children]]
                    Series_ID.append(series.findall('ID')[0].text)
                    Name.append(children.findall('Name')[0].text)
                    AvgThickness.append(children.findall('AvgThickness')[0].text)
                    Volume.append(children.findall('Volume')[0].text)

        col = ['Series_ID', 'Name', 'AvgThickness', 'Volume']
        ThicknessGrid_df = pd.DataFrame([Series_ID, Name, AvgThickness, Volume]).T
        ThicknessGrid_df.columns = col
        return (ThicknessGrid_df)

    ######################################################################################################
    ##next section of class methods provides all table of information necessary for exploratory analysis##
    def get_patient_table(self):
        '''Returns PatientID_DB, Patient_ID, date, sex as pandas df'''
        patient = self.get_patient()
        for child in patient:
            if "Sex" in str(child):
                sex = child.text
            if "ID" in str(child) and "Patient" not in str(child):
                ID = child.text
        PatientID = self.pID_orNAN()

        date = self.get_patient_birthdate()
        d = {'Patient_ID': [ID], 'PatientID_DB': [PatientID], 'sex': [sex], 'date': [date]}
        return (pd.DataFrame(data=d))

    def get_study_table(self):
        '''Returns Patient_ID, Study_ID, study_date as pandas df'''
        PatientID = []
        study_date = self.get_study_date()
        parent_map = self.get_parentmap()
        study = self.get_study()
        for child in study:
            if 'ID' in str(child) and 'Study' not in str(child):
                study_ID = child.text
                patient_id = parent_map[parent_map[child]]
                # print(patient_id)
                PatientID.append(patient_id.findall('ID')[0].text)

        study_d = {'Patient_ID': PatientID, 'Study_ID': [study_ID], 'study_date': [study_date]}
        return (pd.DataFrame(data=study_d))

    def get_series_table(self):
        Modality = []
        Laterality = []
        series_type = []
        NumImages = []
        ModalityProcedure = []
        parent_map = self.get_parentmap()
        study = self.get_study()
        for child in study:
            if 'Series' in str(child):
                for children in child:
                    if 'Modality' in str(children) and 'Procedure' not in str(children):
                        Modality.append(children.text)
                    if 'ModalityProcedure' in str(children):
                        ModalityProcedure.append(children.text)
                    if 'Type' in str(children):
                        series_type.append(children.text)
                    if 'Laterality' in str(children):
                        Laterality.append(children.text)
                    if 'NumImages' in str(children):
                        NumImages.append(children.text)

        series_d = {'Modality': Modality,\
                    'ModalityProcedure': ModalityProcedure, \
                    'series_type': series_type, 'laterality': Laterality, 'NumImages': NumImages}

        series_pd = pd.DataFrame(data=series_d)
        # thickness_pd = get_thicknessgrid(root, tree)

        return (series_pd)

    @property
    def get_image_table(self):
        parent_map = self.get_parentmap()
        series = self.get_series()
        image = self.get_images()
        light_source = []
        series_list = []
        image_list = []
        series = []
        image_aq_time = []
        laterality = []
        type_list = []
        width = []
        height = []
        scaleX = []
        scaleY = []
        startx = []
        starty = []
        endx = []
        endy = []
        oct_width = []
        oct_height = []
        k = 0
        image_id_number = []
        for s in self.get_series():
            #filter for Volume types

            if s.find('Type').text == "Volume":
                for child in s:
                    if child.tag == 'Image':
                        im_id = child.find('ID').text
                        image_id_number.append(int(im_id))
                        for children in child.findall("OphthalmicAcquisitionContext"):

                            # in case of non OCT image, the start end wont we there and we append nan
                            start_check = children.find('Start')
                            if start_check is None:
                                startx.append(np.nan)
                                starty.append(np.nan)

                            end_check = children.find('End')
                            if end_check is None:
                                endx.append(np.nan)
                                endy.append(np.nan)

                            oct_w = children.find('Width').text
                            oct_h = children.find('Height').text


                            oct_width.append(oct_w)
                            oct_height.append(oct_h)

                            for grand_children in children.findall("Width"):
                                width.append(grand_children.text)
                            for grand_children in children.findall("Height"):
                                height.append(grand_children.text)
                            for grand_children in children.findall("ScaleX"):
                                scaleX.append(grand_children.text)
                            for grand_children in children.findall("ScaleY"):
                                scaleY.append(grand_children.text)
                            for grand_children in children.findall("Start"):
                                startx.append(grand_children[0][0].text)
                                starty.append(grand_children[0][1].text)
                            for grand_children in children.findall("End"):
                                endx.append(grand_children[0][0].text)
                                endy.append(grand_children[0][1].text)


                            series_elem = parent_map[child]
                            # add series element for each isntance
                            series.append(str(series_elem))
                            image_id = child[0]
                            for children in child.findall("AcquisitionTime"):
                                time = children[0]

                                for grand_child in time:
                                    # print(grand_child)
                                    if 'Hour' in str(grand_child):
                                        hour = grand_child.text.zfill(2)

                                    if 'Minute' in str(grand_child):
                                        minute = grand_child.text.zfill(2)
                                        # print(minute)
                                    if 'Second' in str(grand_child):
                                        second = grand_child.text
                                        second = str(int(math.floor(float(second)))).zfill(2)

                                image_aq_time.append('-'.join((hour, minute, second)))

                            for children in child.findall("Laterality"):
                                laterality.append(children.text)
                            for children in child.findall("ImageType"):
                                type_list.append(children[0].text)

                            d = {'Image_aq_time': image_aq_time, "series": series, "Laterality":laterality,
                                'Image_type' :type_list,"Width":width, "Height": height, "scaleX": scaleX,
                                "scaleY": scaleY,"oct_width":oct_width, "oct_height": oct_height,
                                "startx_pos":startx, "starty_pos" :starty,"endx_pos": endx,"endy_pos":endy,"image_id":image_id_number}
        return (pd.DataFrame(data = d))

    ##Below all table information are merged into final output##
    def complete_series(self):
        '''returns a merger of all tables containing series information above'''
        t_grid = self.get_thicknessgrid()
        z_grid = self.get_thicknessgrid_zones()
        s_table = self.get_series_table()
        # display(t_grid, z_grid)
        tz_grid = t_grid.merge(z_grid, left_on='Series_ID', right_on='Series_ID', how='left')
        # display(s_table, tz_grid)
        full_series = s_table.merge(tz_grid, left_on='series_ID', right_on='Series_ID', how='left')

        return (full_series)

    def get_explorative_data(self):
        series_complete = self.complete_series()
        study_table = self.get_study_table()
        patient_table = self.get_patient_table()
        # display(study_table,series_complete)
        # series_complete = series_complete.T.drop_duplicates().T
        study_series = study_table.merge(series_complete, left_on='Study_ID', right_on='Study_ID', how='left')
        study_series = study_series.T.drop_duplicates().T
        pat_st_ser = patient_table.merge(study_series, left_on='Patient_ID', right_on='Patient_ID', how='left')
        # pat_st_ser = pat_st_ser.T.drop_duplicates().T
        return (pat_st_ser)


