from feature_statistics.config import WORK_SPACE
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import LSTM.sequences as sequences# <- this contains the custom code

workspace_dir = WORK_SPACE

# longitudinal data is a merged table from all oct measurements and the cleaned diagnosis table
longitudinal_data = pd.read_csv(os.path.join(workspace_dir, 'longitudinal_data.csv'), index_col=0)

# filter measurements for which could not calculate features (from table that Olle sent me)

# if feature stat table exists load here

# join with longitudinal data table


# events is a table containing injections and lens surgery events for each patient
events = pd.read_csv(os.path.join(workspace_dir, 'longitudinal_events.csv'), index_col=0)
events = events.sort_values('study_date')
events.loc[:, 'visus?'] = False
events.loc[:, 'oct?'] = False

# get grouped patients (sorted by date)
# keep NAN octs and logMARs (can still build sequence from them)

# remove patients without cleaned diagnosis label (currently only AMD and DR have diagnosis label)
filtered_diagnosis = longitudinal_data.dropna(subset=['diagnosis'])
filtered_oct_path = filtered_diagnosis.dropna(subset=['oct_path'])
all_patients = filtered_oct_path.sort_values('study_date')

# drop all groups that do not have at least one OCT and one logMAR
grouped = all_patients.groupby(['patient_id', 'laterality'])
all_patients = grouped.filter(lambda x: x.oct_path.count() > 0 and x.logMAR.count() > 0)

grouped_patients = all_patients.groupby(['patient_id', 'laterality'])
grouped_events = events.groupby(['patient_id', 'laterality'])


# create sequences with events added to next mmt
seqs = []
i = 0
for name, group in tqdm(grouped_patients):
    # get events for this group
    group_events = None
    try:
        group_events = grouped_events.get_group(name)
    except KeyError as e:
        pass

    seq = sequences.MeasurementSequence.from_pandas(group)
    seq.add_events_from_pandas(group_events, how='next')  # IMPORTANT: ADD EVENTS TO NEXT MEASUREMENT
    seqs.append(seq)


# parameters for sequence generation
# should each measurement in the sequence have an OCT and a VA?
req_sequence_oct = True
req_sequence_va = False # could just require VA for initial mmt if need more measurements

# do the checkup measurement need to have an OCT and a VA?
# For me not, but for statistics, maybe you need to set req_checkup_oct to True
req_checkup_oct = True
req_checkup_va = False


# create sequences with 3 month / 12 month checkup
sequences_checkup_3 = []
sequences_checkup_3_12 = []
for seq in tqdm(seqs):
    # get seq_ids - all mmts fullfilling criterion
    seq_ids = []
    for seq_id in range(len(seq)):
        # assign measurement its sequence id
        seq.measurements[seq_id].seq_id = seq_id
        if seq.measurements[seq_id].is_valid(req_oct = req_sequence_oct, req_va = req_sequence_va):
            seq_ids.append(seq_id)

    # iterate over all possible end_ids - mmt with has_checkup()
    for i, end_id in enumerate(seq_ids):
        checkup_3_id = seq.has_checkup(end_id, checkup_time = 90, max_deviation = 20,
                                       req_oct = req_checkup_oct, req_va = req_checkup_va)
        checkup_12_id = seq.has_checkup(end_id, checkup_time = 360, max_deviation = 30,
                                        req_oct = req_checkup_oct, req_va = req_checkup_va)
        if checkup_3_id:
            # print(seq_ids[0:i+1]+[checkup_3_id])
            # is valid end_id for 3
            # get new subsetted sequence
            seq_sub = seq.subset(seq_ids[0:i + 1] + [checkup_3_id])
            sequences_checkup_3.append(seq_sub)
            if checkup_12_id:
                # is valid end_id for 3-12
                # get new subsetted sequence
                seq_sub = seq.subset(seq_ids[0:i + 1] + [checkup_3_id, checkup_12_id])
                sequences_checkup_3_12.append(seq_sub)


print("stop here")
