from pprint import pprint
import numpy as np
import pandas as pd
import os
from pathlib import Path


FEATURE_COLUMNS = ['patient_id', 'study_date', 'laterality', 'frame', '0', '1', '10', '11', '12',
                   '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9']


def get_feature_table(feature_table_paths):
    """
    @return: joined feature table dataframes
    @rtype: DataFrame
    """
    feature_table = pd.DataFrame(columns = FEATURE_COLUMNS)
    for path in feature_table_paths:
        table = pd.read_csv(path, index_col = 0)
        feature_table = feature_table.append(table)

    feature_table.study_date = feature_table.study_date.astype(str)
    feature_table.patient_id = feature_table.patient_id.astype(str)
    return feature_table


def apply_feature_filter(features_table):
    """
    @param features_table:
    @type features_table:
    @return:
    @rtype:
    """
    fibrosis_bool = features_table["13"] > 50
    drusen_bool = features_table["8"] > 50
    srhm_bool = features_table["5"] > 50
    # fibro_vasc_bool = features_table["7"] > 50

    features_table_filtered = features_table[fibrosis_bool | drusen_bool | srhm_bool]

    features_table_filtered["frame"] = features_table_filtered.frame.astype("int").copy()
    return features_table_filtered

class Filter():
    def __init__(self, ft_paths, uae_paths):
        # set paths to instances
        self.feature_table_paths = ft_paths
        self.uae_paths = uae_paths

    def selection_table(self):
        """
        @return: joined feature table dataframes
        @rtype: DataFrame
        """
        feature_table = pd.DataFrame(columns = FEATURE_COLUMNS)
        for path in self.feature_table_paths:
            table = pd.read_csv(path, index_col = 0)
            feature_table = feature_table.append(table)

        feature_table.study_date = feature_table.study_date.astype(str)
        feature_table.patient_id = feature_table.patient_id.astype(str)
        return feature_table

    def filter_paths(self, features_table):
        """
        @param features_table:
        @type features_table:
        @return:
        @rtype:
        """
        fibrosis_bool = features_table["13"] > 50
        drusen_bool = features_table["8"] > 50
        srhm_bool = features_table["5"] > 50
        fibro_vasc_bool = features_table["7"] > 50
        
        print(features_table.shape, sum(fibrosis_bool))
        features_table_filtered = features_table[fibrosis_bool | drusen_bool | srhm_bool |fibro_vasc_bool]

        features_table_filtered["frame"] = features_table_filtered.frame.astype("int").copy()
        return features_table_filtered


if __name__ == "__main__":
    import sys

    path = Path(os.getcwd())
    sys.path.append(str(path.parent))
    sys.path.append(str(path.parent.parent))

    from file_manager import FileManager
    import utils
    from utils import Args
    
    args = Args()

    file_manager = FileManager("annotated_files.csv")

    # get record paths
    unannotated_pd, annotated_pd = file_manager.unannotated_records(use_cache = False)
    # unannotated_pd = unannotated_pd.sample(args.number_to_search)
    filter = Filter(file_manager.feature_table_paths, unannotated_pd)

    features_table = filter.selection_table()
    
    print("#"*30) 
    print(features_table.shape, sum(features_table["13"] > 50))
    
    print("#"*30)
    print(unannotated_pd.columns)

    keys = ["patient_id", "laterality", "study_date"]
    features_table_pd = pd.merge(unannotated_pd, features_table, left_on = keys, right_on = keys, how = "left")
    
    print(features_table_pd.shape)
    features_ffiltered_pd = filter.filter_paths(features_table_pd)

    pprint(features_table.head(5))
    pprint(features_ffiltered_pd.head(5))

    print("number of unfiltered samples are:", features_table.shape)
    print("number of filtered samples are:", features_ffiltered_pd.shape)
    print("the columns listed are: ", features_ffiltered_pd.columns)
    assert sum(unannotated_pd.patient_id.isin(annotated_pd.patient_id.values)) == 0, "patient overlap"
    # assert sum(features_ffiltered_pd["13"] < 50) == 0, "all record contains feature oi"
    assert features_table is not None, "returning None"
    assert features_ffiltered_pd is not None, "returning None"
    assert "embedding_path" in features_ffiltered_pd.columns.tolist(), "embedding path not in dataframe"
    assert features_ffiltered_pd.frame.dtype == np.int, "frame dtype is not int"
