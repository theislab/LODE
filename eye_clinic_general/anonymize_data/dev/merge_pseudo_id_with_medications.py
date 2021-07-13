import os
import pandas as pd

'''
General comments:

You need to complete the TODOs below for the script to run. It mainly consists of specifying the names of the 
pseudo id to real id mapping and medication file name. Also the paths to where on the computer these files are located
are needed, see below. 

The last thing you need to make sure is the name of the real id and pseudo id columns in the real to pseudo id mapping
file. THe name of the columns for the real patient id should be written within the '' quotes as instructed below.
'''

# TODO: set path where psuedo id mapping file is here
pseudo_id_directory = "./"

# TODO: set file name of pseudo id mapping file
pseudo_id_file_name = "real_pseudo_id_mapping.csv"

# TODO: set path where medications file is here
medications_directory = "./"

# TODO: set filename of medications table here
medications_file_name = "FAke_example.xlsx"

# TODO: set name of real patient id column name from real to psuedo id mapping file
real_id_mapping_column_name = 'PATNR'
# load id mapping file
id_mapping = pd.read_csv(os.path.join(pseudo_id_directory, pseudo_id_file_name))

# load medications table
medications_table = pd.read_excel(os.path.join(medications_directory, medications_file_name))

# add pseduo id column to medication table
merged_table = pd.merge(medications_table, id_mapping, left_on = "PATNR", right_on = real_id_mapping_column_name,
                        how = "inner")

# delete unecessary columns
merged_table_wo_unecessary_columns = merged_table.drop(columns=["PATNR", 'LNRIC', 'DIAG',
       'HAUPTDIAG', 'NEBENDIAG', 'DIAGKLAU', 'FALNR', 'LFDBEW', 'OPNR',
       'DOKAR', 'DOKNR', 'DOKVR', 'DOKTL', 'DTID', 'MAN', 'THRPNR', 'THRP',
       'DAYSLASTINJ', 'DAYSTHRP', 'VALID', 'DATINDB'])

# save anonymized medication table to be sent to Olle :)
merged_table_wo_unecessary_columns.to_csv("medications_anonymized.csv", index = False)