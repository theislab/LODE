from feature_statistics.config import WORK_SPACE
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import feature_statistics.sequences as sequences

if __name__ == "__main__":
    workspace_dir = WORK_SPACE

    # create sequence data save dir
    save_dir = os.path.join(workspace_dir, 'joint_export/sequence_data')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # longitudinal data is a merged table from all oct measurements and the cleaned diagnosis table
    longitudinal_data = pd.read_csv(os.path.join(workspace_dir, 'joint_export/longitudinal_data/longitudinal_data.csv'), index_col = 0)

    feature_names = None
    segmentation_feature_path = os.path.join(workspace_dir, "joint_export/segmentation/segmentation_statistics.csv")

    # if feature stat table exists load here
    segmented_data = pd.read_csv(segmentation_feature_path, index_col = 0)

    longitudinal_data, feature_names = sequences.check_features(segmented_data, longitudinal_data)

    # change the feature names in measurements
    if feature_names:
        sequences.Measurement.FEATURES = feature_names

    # events is a table containing injections and lens surgery events for each patient
    events = pd.read_csv(os.path.join(workspace_dir, 'joint_export/longitudinal_data/longitudinal_events.csv'), index_col=0)
    events = events.sort_values('study_date')
    events.loc[:, "study_date"] = pd.to_datetime(events.study_date).dt.date.astype(str)

    # set string NaT to np.nan
    # events.study_date = events.study_date.replace("NaT", np.nan)

    events.loc[:, 'visus?'] = False
    events.loc[:, 'oct?'] = False

    # remove patients without cleaned diagnosis label (currently only AMD and DR have diagnosis label)
    filtered_diagnosis = longitudinal_data.dropna(subset = ['diagnosis'])
    filtered_oct_path = filtered_diagnosis.dropna(subset = ['oct_path'])
    all_patients = filtered_oct_path.sort_values('study_date')

    # all_patients = all_patients.loc[filtered_diagnosis.patient_id == 378649]

    # drop all groups that do not have at least one OCT and one logMAR
    grouped = all_patients.groupby(['patient_id', 'laterality'])
    all_patients = grouped.filter(lambda x: x.oct_path.count() > 0 and x.logMAR.count() > 0)

    grouped_patients = all_patients.groupby(['patient_id', 'laterality'])
    grouped_events = events.groupby(['patient_id', 'laterality'])

    # create sequences with events added to next mmt
    seqs = []
    i = 0
    for name, group in tqdm(grouped_patients):
        if name == (32179, 'R'):

            # get events for this group
            group_events = None
            try:
                group_events = grouped_events.get_group(name)
            except KeyError as e:
                pass

            seq = sequences.MeasurementSequence.from_pandas(group)
            seq.add_events_from_pandas(group_events, how = 'next')  # IMPORTANT: ADD EVENTS TO NEXT MEASUREMENT
            seqs.append(seq)

    # parameters for sequence generation
    # should each measurement in the sequence have an OCT and a VA?
    req_sequence_oct = True
    req_sequence_va = False  # could just require VA for initial mmt if need more measurements

    # do the checkup measurement need to have an OCT and a VA?
    # For me not, but for statistics, maybe you need to set req_checkup_oct to True
    req_checkup_oct = True
    req_checkup_va = False

    # create sequences with 3 month / 12 month checkup
    sequences_checkup = []
    for seq in tqdm(seqs):
        # get seq_ids - all mmts full-filling criterion
        seq_ids = []
        for seq_id in range(len(seq)):
            # assign measurement its sequence id
            seq.measurements[seq_id].seq_id = seq_id
            if seq.measurements[seq_id].is_valid(req_oct = req_sequence_oct, req_va = req_sequence_va):
                seq_ids.append(seq_id)

        # iterate over all possible end_ids - mmt with has_checkup() - set min(seq_ids) t0 just consider first visit
        for i, end_id in enumerate([min(seq_ids)]):
            # sequence must be at least two visits
            if len(seq.measurements) > 1:
                checkup_full_id = np.max(seq_ids)
                seq_sub = seq.subset(seq_ids[0:i + 1] + [checkup_full_id], keep_followup = True)
                sequences_checkup.append(seq_sub)

    # save sequences to file to avoid recomputing
    sequences.save_sequences_to_dataframe(os.path.join(save_dir, 'sequences.csv'), sequences_checkup)
