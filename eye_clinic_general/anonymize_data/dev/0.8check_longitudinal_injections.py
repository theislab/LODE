import os
import pandas as pd

dwh_tables_dir = "/media/basani/Seagate Expansion Drive/2_lists"
temp_tables_dir = "/media/basani/Seagate Expansion Drive/temp_tables"

previous_export = "pseudo_id_key.csv"
id_mapping = "pseudo_real_id_mapping.csv"
table = "./2_lists/prozeduren.csv"


id_mapping_table = pd.read_csv(os.path.join(temp_tables_dir, id_mapping))
id_mapping_table = id_mapping_table.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])


prozeduren = pd.read_csv(table)

# filter for patients with injections
prozeduren_w_injections = prozeduren[prozeduren.ICPML == "5-156.9"]
prozeduren_w_injections = prozeduren_w_injections[["PATNR", "ICPML", "LOK", "DAT"]]

prozeduren_w_injections["DAT_dt"] = pd.to_datetime(prozeduren_w_injections.DAT)

prozeduren_w_injections["DAT_DIFF"] = prozeduren_w_injections.sort_values(['PATNR', 'DAT_dt']).groupby('PATNR')['DAT_dt'].diff(1)

# drop time diff na
prozeduren_w_injections = prozeduren_w_injections[~prozeduren_w_injections["DAT_DIFF"].isna()]


all_patients = prozeduren_w_injections[["PATNR"]].drop_duplicates()

# convert days to ints
all_patients["DAT_SPAN"] = prozeduren_w_injections.groupby('PATNR')['DAT_DIFF'].sum().values
all_patients = all_patients[all_patients.DAT_SPAN.dt.days > 90]

prev_export_table = pd.read_csv(os.path.join(dwh_tables_dir, previous_export), sep=";")

# filter old patients
prev_export_filter = all_patients.PATNR.isin(prev_export_table.PATNR.values.tolist())

longitudinal_table_only_current_export = all_patients[~prev_export_filter]

longitudinal_table_only_current_export = longitudinal_table_only_current_export[["PATNR"]].drop_duplicates()


# merge export table
inj_export_table = pd.merge(longitudinal_table_only_current_export, id_mapping_table, left_on="PATNR",
                            right_on="patient_id", how="left")

inj_export_table = inj_export_table[inj_export_table.modality == "OPT"]