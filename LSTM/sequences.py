import pickle
from datetime import datetime
import numpy as np
from copy import deepcopy
from LSTM.datasets import IOVar
import os
from tqdm import tqdm
import pandas as pd


def save_sequences_to_pickle(fname, sequences):
    """Serialize sequences to pickle format"""
    d = [seq.to_dict() if isinstance(seq, MeasurementSequence) else seq for seq in sequences]
    pickle.dump(d, open(fname, 'wb'))


def save_sequences_to_dataframe(fname, sequences):
    """convert sequences to dataframe format"""
    d = [seq.to_dataframe() if isinstance(seq, MeasurementSequence) else seq for seq in sequences]

    main_dataframe = d[0]
    # create main data frame
    for i in tqdm(range(1, len(d))):
        main_dataframe = main_dataframe.append(d[i])

    # save data frame
    main_dataframe.to_csv(fname, columns = main_dataframe.columns, index = False)


def load_sequences_from_pickle(fname):
    """Load sequences from serialized pickle file"""
    d_list = pickle.load(open(fname, 'rb'))
    sequences = []
    for d in d_list:
        if isinstance(d, bool):
            sequences.append(d)
        else:
            sequences.append(MeasurementSequence.from_dict(d))
    return sequences


def split_sequences(sequences, min_len=None, max_len=None, train_frac=0.8, val_frac=0.1, num_train=None, diagnosis=None,
                    seed=42, log=None, only_hard_sequences=False):
    """define train/val/test splits of all sequences with length sequence_length:
        1. restrict sequences to correct length (using min_len and max_len)
        2. (optional) restrict input sequences to diagnosis
        3. (optional) restrict input sequences to certain class (only_hard_sequences flag)
        4. shuffle sequences with seed
        5. split train_frac and val_frac sequences, the rest goes to test
        6. (optional) restrict train to num_train, and adjust val and test accordingly
    
    Returns:
        sequences_train, sequences_val, sequences_test
    """
    # 1. restrict input sequences to correct length
    if min_len is None:
        min_len = 1
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    cur_sequences = [seq for seq in sequences if len(seq) >= min_len and len(seq) <= max_len]

    # 2. restrict input sequences to diagnosis
    if diagnosis is not None:
        cur_sequences = [seq for seq in cur_sequences if seq.diagnosis == diagnosis]

    # 3. restrict input sequences to certain class
    if only_hard_sequences is True:
        print('restricting to only hard sequences!')
        # only those sequences that have NOT label 2 (no change)
        cur_sequences = [seq for seq in cur_sequences if
                         np.argmax(IOVar.CHECKUP1_DIFF_VA_CLASS.get_data_from_mmt(seq.measurements[-1])) != 2]

    # 3. shuffle sequences with seed
    # get all patient ids
    patient_ids, indices = np.unique(np.array(sorted([seq.patient_id for seq in cur_sequences])), return_inverse = True)
    N_pat = len(patient_ids)
    # split in train / test / val
    np.random.seed(41)
    ordering = np.arange(0, N_pat)
    np.random.shuffle(ordering)

    # 4. split train_frac and val_frac sequences, the rest goes to test
    N_pat_train = int(round(train_frac * N_pat))
    N_pat_val = int(round(val_frac * N_pat))

    sequences_train = np.array(cur_sequences)[np.isin(indices, ordering[:N_pat_train])]
    sequences_val = np.array(cur_sequences)[np.isin(indices, ordering[N_pat_train:N_pat_train + N_pat_val])]
    sequences_test = np.array(cur_sequences)[np.isin(indices, ordering[N_pat_train + N_pat_val:])]

    np.random.shuffle(sequences_train)
    np.random.shuffle(sequences_val)
    np.random.shuffle(sequences_test)

    # 5. (optional) restrict train to num_train, and adjust val and test accordingly
    if num_train is not None:
        num_val = int(round(num_train / train_frac * val_frac))
        num_test = int(round(num_train / train_frac * (1 - train_frac - val_frac)))
        sequences_train = sequences_train[:num_train]
        sequences_val = sequences_val[:num_val]
        sequences_test = sequences_test[:num_test]

    log_str = 'Split sequences in {} train ({:.01f}%), {} val ({:.01f}%), {} test ({:.01f}%)'.format(
        len(sequences_train), len(sequences_train) / len(cur_sequences) * 100,
        len(sequences_val), len(sequences_val) / len(cur_sequences) * 100,
        len(sequences_test), len(sequences_test) / len(cur_sequences) * 100)
    if log is not None:
        log.info(log_str)
    else:
        print(log_str)

    return sequences_train, sequences_val, sequences_test


def check_features(workspace_dir, longitudinal_data):
    """
    workspace_dir: str
    longitudinal_data: DataFrame with long. data
    """
    feature_names = None
    segmentation_feature_path = os.path.join(workspace_dir, "sequence_data/segmentation_statistics.csv")

    assert os.path.exists(segmentation_feature_path), "Features not available in work space"

    # if feature stat table exists load here
    segmented_data = pd.read_csv(segmentation_feature_path, index_col = 0)

    # get feature names
    feature_names = segmented_data.columns[1:]

    # join with longitudinal data table
    keys = ["patient_id", "laterality", "study_date"]
    segmented_data[keys] = segmented_data.record.str.split("_", expand = True)[[0, 1, 2]]

    # convert keys to same format is in longitudinal_data
    segmented_data["study_date"] = pd.to_datetime(segmented_data["study_date"]).astype(str)

    # convert patient id to int
    segmented_data["patient_id"] = segmented_data["patient_id"].astype(np.int64)

    longitudinal_data = pd.merge(longitudinal_data, segmented_data, left_on = keys, right_on = keys, how = "inner")
    return longitudinal_data, feature_names.tolist()


class Measurement:
    MEDS = ['Avastin', 'Dexamethason', 'Eylea', 'Iluvien', 'Jetrea', 'Lucentis', 'Ozurdex', 'Triamcinolon', 'Unknown']
    FEATURES = []

    # -- Initializers --
    def __init__(self, study_date, oct_path, cur_va, table, seq_id):
        if isinstance(study_date, datetime):
            self.study_date = study_date
        else:
            self.study_date = datetime.strptime(study_date, '%Y-%m-%d')

        if oct_path is None or str(oct_path) == 'nan':
            self.oct_path = None
        else:
            self.oct_path = str(oct_path)

        if cur_va is None or str(float(cur_va)) == 'nan':
            self.cur_va = None
        else:
            self.cur_va = float(cur_va)

        # other Measurement attributes not set in __init__
        self.delta_t = None
        self.next_va = None
        self.injections = [0 for _ in Measurement.MEDS]
        self.injection_dates = [np.nan for _ in Measurement.MEDS]

        # if features are available, C0_total is example feature
        if "C0_total" in table.index.tolist():
            self.features = [table[feature] for feature in Measurement.FEATURES]
        else:
            self.features = 0
        self.lens_surgery = False
        self.seq_id = seq_id

    @classmethod
    def set_features(cls, features):
        FEATURES = features

    @classmethod
    def from_pandas(cls, row, seq_id):
        return cls(row.study_date, row.oct_path, row.logMAR, row, seq_id)

    @classmethod
    def from_dict(cls, d):
        """initialize from serialized dict"""
        self = cls(d['study_date'], d['oct_path'], d['cur_va'],  d["features"].loc[0], d["seq_id"])
        self.delta_t = d['delta_t']
        self.lens_surgery = d['lens_surgery']
        self.injections = d['injections']
        self.next_va = d['next_va']
        return self

    def __str__(self):
        oct_path = self.oct_path is not None
        cur_va = '{:.2f}'.format(self.cur_va) if self.cur_va is not None else None
        next_va = '{:.2f}'.format(self.next_va) if self.next_va is not None else None
        delta_t = '{:04}'.format(self.delta_t) if self.delta_t is not None else None
        res = 'Measurement {:%Y-%m-%d}: oct {:5}, cur_va {}, delta_t {}, next_va {}, {} injections, lens_surgery {:5}'.format(
            self.study_date, str(oct_path), cur_va, delta_t, next_va, sum(self.injections), str(self.lens_surgery))
        return res

    # -- methods --
    def set_next_values(self, mmt):
        """set next_va and delta_t from following Measurement object"""
        self.next_va = mmt.cur_va
        self.delta_t = (mmt.study_date - self.study_date).days

    def add_event_from_pandas(self, event):
        """add lens surgery or injection event"""
        # evt_date = datetime.strptime(event.study_date, '%Y-%m-%d')
        # assert evt_date >= self.study_date, "event date cannot be earlier than measurement date!"

        if event['iol?'] is True:
            self.lens_surgery = True

        if event['injection?'] == True:
            if event[['MED']].isna().iloc[0]:
                event['MED'] = "Unknown"

            self.injections[Measurement.MEDS.index(event['MED'])] += 1
            self.injection_dates.append(datetime.strptime(event['study_date'], '%Y-%m-%d'))

    def is_valid(self, req_va=False, req_oct=False):
        """
        Check if a measurement fulfills the given criterions
        """
        has_va = self.cur_va is not None
        has_oct = self.oct_path is not None
        return (not req_va or has_va) and (not req_oct or has_oct)

    def to_dict(self):
        """serialize as dict"""
        d = {
            'study_date': '{:%Y-%m-%d}'.format(self.study_date),
            'oct_path': self.oct_path,
            'cur_va': self.cur_va,
            'delta_t': self.delta_t,
            'next_va': self.next_va,
            'injections': self.injections,
            'lens_surgery': self.lens_surgery,
            'seq_id': self.seq_id,
            'features': pd.DataFrame([self.features], columns=Measurement.FEATURES)
        }
        return d

    def to_dataframe(self):
        """serialize as dict"""
        d = {
            'study_date': '{:%Y-%m-%d}'.format(self.study_date),
            'oct_path': self.oct_path,
            'cur_va': self.cur_va,
            'delta_t': self.delta_t,
            'next_va': self.next_va,
            'injections': sum(self.injections),
            'lens_surgery': self.lens_surgery,
            'seq_id': self.seq_id,
        }

        # add all injections
        for k, med in enumerate(Measurement.MEDS):
            d[f"injection_{med}"] = self.injections[k]
            #for j in range(self.injections[k] + k):
            #    d[f"injection_date_{med}"] = self.injection_dates[j]

        # add all clinical features
        for i, feature in enumerate(Measurement.FEATURES):
            d[feature] = self.features[i]
        return d


class MeasurementSequence:

    def __init__(self, diagnosis, measurements, patient_id, laterality):
        self.diagnosis = diagnosis
        self.measurements = measurements
        self.patient_id = patient_id
        self.laterality = laterality
        self.injections_before_oct = 0

        # properties (calculated)
        self.__num_cur_va = None
        self.__num_next_va = None

    @classmethod
    def from_pandas(cls, mmt_table):
        diagnosis = mmt_table.diagnosis.iloc[0]
        patient_id = int(mmt_table.patient_id.iloc[0])
        laterality = mmt_table.laterality.iloc[0]

        # set first measurement observed
        measurements = [Measurement.from_pandas(mmt_table.iloc[0], seq_id = 0)]

        for i in range(1, len(mmt_table)):
            mmt = Measurement.from_pandas(mmt_table.iloc[i], seq_id = i)
            measurements[-1].set_next_values(mmt)

            # set sequence id
            measurements[-1].seq_id = i
            if mmt.oct_path is not None:
                measurements.append(mmt)
        return cls(diagnosis, measurements, patient_id, laterality)

    @classmethod
    def from_dict(cls, d):
        measurements = [Measurement.from_dict(mmt) for mmt in d['measurements']]
        return cls(diagnosis = d['diagnosis'], measurements = measurements, patient_id = d['patient_id'],
                   laterality = d.get('laterality', None))

    def __len__(self):
        return len(self.measurements)

    @property
    def num_cur_va(self):
        if self.__num_cur_va is None:
            num_va = 0
            for mmt in self.measurements:
                if mmt.cur_va is not None:
                    num_va += 1
            self.__num_cur_va = num_va
        return self.__num_cur_va

    @property
    def num_next_va(self):
        if self.__num_next_va is None:
            num_va = 0
            for mmt in self.measurements:
                if mmt.next_va is not None:
                    num_va += 1
            self.__num_next_va = num_va
        return self.__num_next_va

    @property
    def num_injections(self):
        num = 0
        for mmt in self.measurements:
            num += sum(mmt.injections)
        return num

    @property
    def lens_surgery(self):
        for mmt in self.measurements:
            if mmt.lens_surgery:
                return True
        return False

    def __str__(self):
        res = 'MeasurementSequence {},{} ({}): [\n'.format(self.patient_id, self.laterality, len(self.measurements))
        for mmt in self.measurements:
            res += str(mmt)
            res += '\n'
        res += ']'
        return res

    # -- methods --  
    def add_events_from_pandas(self, events_table, how='previous'):
        """adds medications and lens_surgery events to measurements objects
        events e with mmt1.study_date <= e.study_date < mmt2.study_date is added to mmt1
        Args:
            how: previous: add events to previous mmt,
                 next: add events to next mmt
        """
        if events_table is None:
            return

        # events are ordered, thus mmt_id will be increasing
        mmt_id = 0
        evt_id = 0
        while (mmt_id < len(self.measurements) + 1) and (evt_id < len(events_table)):
            evt = events_table.iloc[evt_id]
            evt_date = datetime.strptime(evt.study_date, '%Y-%m-%d')
            while mmt_id < len(self.measurements) and evt_date >= self.measurements[mmt_id].study_date:
                mmt_id += 1
            # have a mmt_id for which evt_date < self.measurements[mmt_id].study_date
            # assign event to this ore prevoius id depending on `how`
            if how == 'next':
                if mmt_id == len(self.measurements):
                    # event happened after last measuremment: disregard it
                    pass
                else:
                    self.measurements[mmt_id].add_event_from_pandas(evt)
            if how == 'previous':
                if mmt_id == 0:
                    # event happened before first measurement: add it to count
                    self.injections_before_oct += 1
                    self.measurements[mmt_id].add_event_from_pandas(evt)
                else:
                    self.measurements[mmt_id - 1].add_event_from_pandas(evt)

            evt_id += 1

    def add_events_from_pandas_old(self, events_table):
        """adds medications and lens_surgery events to measurements objects
        events e with mmt1.study_date <= e.study_date < mmt2.study_date is added to mmt1
        Args:
            how: previous: add events to previous mmt,
                 next: add events to next mmt
        """
        if events_table is None:
            return

        mmt_id = 0
        # events are ordered, thus mmt_id will be increasing      
        for i in range(0, len(events_table)):
            evt = events_table.iloc[i]
            evt_date = datetime.strptime(evt.study_date, '%Y-%m-%d')
            for j in range(mmt_id, len(self.measurements) + 1):
                if j == len(self.measurements):
                    mmt_id = j - 1
                    break
                if evt_date < self.measurements[j].study_date:
                    mmt_id = j - 1
                    break
            if mmt_id == -1:
                # event happened before first measurement: disregard it
                mmt_id = 0
                continue
            # write event data to selected measurement
            self.measurements[mmt_id].add_event_from_pandas(evt)

    def remove_measurement(self, mmt_id, event_assignment='next'):
        """remove measurement with mmt_id from this sequence.
        Args:
            event_assignment (str): next or previous. Determines how events from the deleted measurement are assigned.
                'next' assigns them to the next measurement in the sequence
                'previous' assigns them to the previous measurement in the sequence.
        """
        # assign events to next measurement
        mmt = self.measurements[mmt_id]
        if mmt_id > 0:
            prev_mmt = self.measurements[mmt_id - 1]
        else:
            prev_mmt = None
        if mmt_id < len(self.measurements) - 1:
            next_mmt = self.measurements[mmt_id + 1]
        else:
            next_mmt = None

        # set next va of prev mmt to va of following mmt
        # set deltat for prev mmt to point to following mmt
        if prev_mmt is not None:
            if next_mmt is not None:
                # set next va of prev mmt to va of following mmt
                prev_mmt.next_va = next_mmt.cur_va
                prev_mmt.delta_t += mmt.delta_t
            else:
                prev_mmt.next_va = None
                prev_mmt.delta_t = None

        # add events to correct measurement
        evt_mmt = None
        if event_assignment == 'next':
            evt_mmt = next_mmt
        elif event_assignment == 'previous':
            evt_mmt = prev_mmt
        if evt_mmt is not None:
            for i, inj in enumerate(mmt.injections):
                evt_mmt.injections[i] += inj
                evt_mmt.injection_dates.extend(mmt.injection_dates)
            if mmt.lens_surgery:
                evt_mmt.lens_surgery = True

        # reset calculated properties
        self.__num_cur_va = None
        self.__num_next_va = None

        # remove mmt from self.measurements
        del self.measurements[mmt_id]

    def subset(self, ids, keep_followup):
        """return new subsetted sequences only containing measurements at ids"""
        seq_sub = deepcopy(self)
        if not keep_followup:
            ids_to_remove = np.setdiff1d(range(0, len(seq_sub)), ids)
        else:
            ids_to_remove = list(filter(lambda x: x > max(ids), list(range(0, len(seq_sub)))))

        ids_to_remove = np.sort(ids_to_remove)[::-1]
        for mmt_id in ids_to_remove:
            seq_sub.remove_measurement(mmt_id)
        return seq_sub

    def has_checkup(self, mmt_id, checkup_time=90, max_deviation=20, req_va=False, req_oct=False):
        """
        Does the given measurement have a checkup `checkup_time` days after the measurement?
        Args:
            mmt_id: index of the measurement 
            checkup_time: time delta in days the checkup should take place after the measurement
            max_deviation: max number of days the actual checkup time may deviate from the desired number of days
            req_va: require VA for checkup
            req_oct: requence OCT for checkup
        """
        if mmt_id == len(self.measurements) - 1:
            # is the last measurement, no checkup possible!
            return False

        delta_t = np.cumsum([mmt.delta_t for mmt in self.measurements[mmt_id:]][:-1])
        deviation_from_checkup_time = np.abs(delta_t - checkup_time)
        # get most probable checkup candidate
        for idx in np.argsort(deviation_from_checkup_time):
            if deviation_from_checkup_time[idx] <= max_deviation:
                # check if has va/oct as required
                if self.measurements[mmt_id + idx + 1].is_valid(req_va = req_va, req_oct = req_oct):
                    # have found checkup, return its mmt_id
                    return mmt_id + idx + 1
            else:
                # can break loop, will not find another candidate
                break
        return False

    def to_dict(self):
        d = {
            'diagnosis': self.diagnosis,
            'patient_id': self.patient_id,
            'laterality': self.laterality,
            'measurements': [mmt.to_dict() for mmt in self.measurements]
        }
        return d

    def to_dataframe(self):
        d = [mmt.to_dataframe() for mmt in self.measurements]
        if d:
            s = pd.DataFrame(columns = list(d[0].keys()))

            # convert to data frame
            for dict_ in d:
                s = s.append(pd.DataFrame.from_dict([dict_]))

            # add sequence meta data
            meta_columns = {"laterality": self.laterality, "diagnosis": self.diagnosis, "patient_id": self.patient_id}
            for meta_str in meta_columns.keys():
                s.insert(loc = 0, column = meta_str, value = meta_columns[meta_str])
            return s
        else:
            return d

    def add_features_from_pandas(self, grouped_features, normalize=None):
        """add feature values to measurement objects. Merge features_table with measurements"""
        for mmt in self.measurements:
            # get features for individual measurement
            try:
                mmt.features = np.array(grouped_features.loc[mmt.oct_path])
                if normalize == 'retina':
                    # normalize by retinal features
                    mmt.features = mmt.features / mmt.features[[1, 4, 5, 6, 7, 8, 9, 10, 11]].sum()
            except KeyError as e:
                print(f'Did not find features for oct {mmt.oct_path}')
            # get features for any checkup measurements
            for checkup in ('checkup1', 'checkup2'):
                checkup_mmt = getattr(mmt, checkup, None)
                if checkup_mmt is not None:
                    try:
                        checkup_mmt.features = np.array(grouped_features.loc[checkup_mmt.oct_path])
                        if normalize == 'retina':
                            # normalize by retinal features
                            checkup_mmt.features = checkup_mmt.features / checkup_mmt.features[
                                [1, 4, 5, 6, 7, 8, 9, 10, 11]].sum()
                    except KeyError as e:
                        print(f'Did not find features for oct {checkup_mmt.oct_path}')
