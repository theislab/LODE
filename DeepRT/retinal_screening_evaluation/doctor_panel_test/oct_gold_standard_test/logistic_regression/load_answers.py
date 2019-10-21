import pandas as pd
def load_answers(fundus_answ_paths, gold_standard):
    # mappings
    abt = pd.read_csv( "./mappings/mapping_fundus_prediction.csv" )

    # start loading all evaluator answers
    answ_1 = pd.read_csv( fundus_answ_paths[0], sep = "," )
    answ_1["e/a/n"] = answ_1["e/a/n"].str.lower().replace( "a ", "a" ).replace( " n", "n" ).replace( " a", "a" )
    answ_1 = answ_1[["id", "e/a/n"]]

    answ_2 = pd.read_csv( fundus_answ_paths[1], sep = "," )
    answ_2["e/a/n"] = answ_2["e/a/n"].str.lower().replace( "a ", "a" ).replace( " n", "n" ).replace( " a", "a" )
    answ_2 = answ_2[["id", "e/a/n"]]

    answ_3 = pd.read_csv( fundus_answ_paths[2], sep = ";" )
    answ_3["e/a/n"] = answ_3["e/a/n"].str.lower().replace( "a ", "a" ).replace( " n", "n" ).replace( " a", "a" )
    answ_3 = answ_3[["id", "e/a/n"]]

    answ_4 = pd.read_csv( fundus_answ_paths[3], sep = "," )
    answ_4["e/a/n"] = answ_4["e/a/n"].str.lower().replace( "a ", "a" ).replace( " n", "n" ).replace( " a", "a" )
    answ_4 = answ_4[["id", "e/a/n"]]

    answ_5 = pd.read_csv( fundus_answ_paths[4], sep = "," )
    answ_5["e/a/n"] = answ_5["e/a/n"].str.lower().replace( "a ", "a" ).replace( " n", "n" ).replace( " a", "a" )
    answ_5 = answ_5[["id", "e/a/n"]]

    answ_6 = pd.read_csv( fundus_answ_paths[5], sep = "," )
    answ_6["e/a/n"] = answ_6["e/a/n"].str.lower().replace( "a ", "a" ).replace( " n", "n" ).replace( " a", "a" )
    answ_6 = answ_6[["id", "e/a/n"]]

    answ_7 = pd.read_csv( fundus_answ_paths[6], sep = "," )
    answ_7["e/a/n"] = answ_7["e/a/n"].str.lower().replace( "a ", "a" ).replace( " n", "n" ).replace( " a", "a" )
    answ_7 = answ_7[["id", "e/a/n"]]

    # gather all answers in list
    answers = [answ_1, answ_2, answ_3, answ_4, answ_5, answ_6, answ_7]

    # merge ansers
    for iter_, answ in enumerate( answers ):
        abt = pd.merge( abt, answ, left_on = "pred_pseudo", right_on = "id", how = "inner" )

    # drop unnecessary columns
    abt = abt.drop( columns = ["id_x", "id_y", "id", "pred_pseudo", "Unnamed: 0"] )

    # number the evaluators
    for i in range( 1, 8 ):
        abt[str( i )] = abt.iloc[:, i]

    # drop previous columns
    abt = abt.drop( columns = ["e/a/n_x", "e/a/n_y", "e/a/n_x", "e/a/n_y", "e/a/n_x", "e/a/n_y", "e/a/n"] )

    # make table long
    long_abt = pd.melt( abt,
                        id_vars = ['record_name'],
                        value_vars = ["1", "2", "3", "4", "5", "6", "7"],
                        var_name = 'doctor',
                        value_name = "answer" )

    # merge with gold standard
    data = pd.merge( gold_standard, long_abt, left_on = "id", right_on = "record_name", how = "inner" )

    # rename and drop columns
    data = data.rename( columns = {"A/N/E_400": "y", "id": "record_id"} )
    data = data.drop( columns = {"record_name"} )

    # make labels categorical
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    data["answer"] = le.fit_transform( data["answer"] )
    data["y"] = le.fit_transform( data["y"] )
    data["record_id"] = le.fit_transform( data["record_id"] )

    return data
