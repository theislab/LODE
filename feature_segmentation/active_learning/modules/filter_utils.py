import pandas as pd

FEATURE_COLUMNS = ['patient_id', 'study_date', 'laterality', 'frame', '0', '1', '10', '11', '12',
                   '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9']


def set_id_columns(features_table):
    # create or overwrite allready correct id
    features_table["id"] = features_table.patient_id.astype(str) + "_" + \
                           features_table.laterality.astype(str) + "_" + \
                           features_table.study_date.astype(str) + "_" + \
                           features_table.frame.astype(str)
    return features_table


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
