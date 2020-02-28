import pickle
from datetime import datetime
import numpy as np

def save_sequences_to_pickle(fname, sequences):
    """Serialize sequences to pickle format"""
    d = [seq.to_dict() for seq in sequences]
    pickle.dump(d, open(fname, 'wb'))
    
def load_sequences_from_pickle(fname):
    """Load sequences from serialized pickle file"""
    d_list = pickle.load(open(fname, 'rb'))
    sequences = []
    for d in d_list:
        sequences.append(MeasurementSequence.from_dict(d))
    return sequences

def split_sequences(sequences, sequence_length, train_frac=0.8, val_frac=0.1, num_train=None, diagnosis=None, seed=42, log=None):
    """define train/val/test splits of all sequences with length sequence_length:
        1. restrict sequences to correct length
        2. (optional) restrict input sequences to diagnosis
        3. shuffle sequences with seed
        4. split train_frac and val_frac sequences, the rest goes to test
        5. (optional) restrict train to num_train, and adjust val and test accordingly
    
    Returns:
        sequences_train, sequences_val, sequences_test
    """
    # 1. restrict input sequences to correct length
    cur_sequences = [seq for seq in sequences if len(seq)==sequence_length]
    
    # 2. restrict input sequences to diagnosis
    if diagnosis is not None:
        cur_sequences = [seq for seq in cur_sequences if seq.diagnosis==diagnosis]
    
    # 3. shuffle sequences with seed
    # get all patient ids
    patient_ids, indices = np.unique(np.array(sorted([seq.patient_id for seq in cur_sequences])), return_inverse=True)
    N_pat = len(patient_ids)
    # split in train / test / val
    np.random.seed(41)
    ordering = np.arange(0, N_pat)
    np.random.shuffle(ordering)
    
    # 4. split train_frac and val_frac sequences, the rest goes to test
    N_pat_train = int(round(train_frac * N_pat))
    N_pat_val = int(round(val_frac * N_pat))

    sequences_train = np.array(cur_sequences)[np.isin(indices, ordering[:N_pat_train])]
    sequences_val = np.array(cur_sequences)[np.isin(indices, ordering[N_pat_train:N_pat_train+N_pat_val])]
    sequences_test = np.array(cur_sequences)[np.isin(indices, ordering[N_pat_train+N_pat_val:])]
    
    np.random.shuffle(sequences_train)
    np.random.shuffle(sequences_val)
    np.random.shuffle(sequences_test)
    
    # 5. (optional) restrict train to num_train, and adjust val and test accordingly
    if num_train is not None:
        num_val = int(round(num_train / train_frac * val_frac))
        num_test = int(round(num_train / train_frac * (1-train_frac-val_frac)))
        sequences_train = sequences_train[:num_train]
        sequences_val = sequences_val[:num_val]
        sequences_test = sequences_test[:num_test]
        
    log_str = 'Split sequences in {} train ({:.01f}%), {} val ({:.01f}%), {} test ({:.01f}%)'.format(
        len(sequences_train), len(sequences_train)/len(cur_sequences)*100,
        len(sequences_val), len(sequences_val)/len(cur_sequences)*100,
        len(sequences_test), len(sequences_test)/len(cur_sequences)*100)
    if log is not None:
        log.info(log_str)
    else:
        print(log_str)
    
    return sequences_train, sequences_val, sequences_test
        

class Measurement:
    MEDS = ['Avastin', 'Dexamethas', 'Eylea', 'Iluvien', 'Jetrea', 'Lucentis', 'Ozurdex', 'Triamcinol']
    # -- Initializers --
    def __init__(self, study_date, oct_path, cur_va):
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
        self.lens_surgery = False
        
    @classmethod
    def from_pandas(cls, row):
        return cls(row.study_date, row.oct_path, row.logMAR)
    
    @classmethod
    def from_dict(cls, d):
        """initialize from serialized dict"""
        self = cls(d['study_date'], d['oct_path'], d['cur_va'])
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
        #evt_date = datetime.strptime(event.study_date, '%Y-%m-%d')
        #assert evt_date >= self.study_date, "event date cannot be earlier than measurement date!"
        
        if event['iol?'] is True:
            self.lens_surgery = True
        if event['injection?'] is True:
            self.injections[Measurement.MEDS.index(event['MED'])] += 1
            
    def to_dict(self):
        """serialize as dict"""
        d = {
            'study_date': '{:%Y-%m-%d}'.format(self.study_date),
            'oct_path': self.oct_path,
            'cur_va': self.cur_va,
            'delta_t': self.delta_t,
            'next_va': self.next_va,
            'injections': self.injections,
            'lens_surgery': self.lens_surgery
        }
        return d


class MeasurementSequence:
    
    def __init__(self, diagnosis, measurements, patient_id):
        self.diagnosis = diagnosis
        self.measurements = measurements
        self.patient_id = patient_id
        
        # properties (calculated)
        self.__num_cur_va = None
        self.__num_next_va = None  
        
    @classmethod
    def from_pandas(cls, mmt_table):
        diagnosis = mmt_table.diagnosis.iloc[0]
        patient_id = int(mmt_table.patient_id.iloc[0])
        measurements = [Measurement.from_pandas(mmt_table.iloc[0])]
        for i in range(1, len(mmt_table)):
            mmt = Measurement.from_pandas(mmt_table.iloc[i])
            measurements[-1].set_next_values(mmt)
            if mmt.oct_path is not None:
                measurements.append(mmt)
                
        return cls(diagnosis, measurements, patient_id)
    
    @classmethod
    def from_dict(cls, d):
        measurements = [Measurement.from_dict(mmt) for mmt in d['measurements']]
        return cls(diagnosis=d['diagnosis'], measurements=measurements, patient_id=d['patient_id'])
    
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
        res = 'MeasurementSequence {} ({}): [\n'.format(self.patient_id, len(self.measurements))
        for mmt in self.measurements:
            res += str(mmt)
            res += '\n'
        res += ']'
        return res
    
        
    # -- methods --  
    def add_events_from_pandas(self, events_table):
        """adds medications and lens_surgery events to measurements objects
        events e with mmt1.study_date <= e.study_date < mmt2.study_date is added to mmt1
        """
        if events_table is None:
            return
        
        # events are ordered, thus mmt_id will be increasing
        mmt_id = 0
        for i in range(0, len(events_table)):
            evt = events_table.iloc[i]
            evt_date = datetime.strptime(evt.study_date, '%Y-%m-%d')
            for j in range(mmt_id, len(self.measurements)+1):
                if j == len(self.measurements):
                    mmt_id = j-1
                    break
                if evt_date < self.measurements[j].study_date:
                    mmt_id = j-1
                    break
            if mmt_id == -1:
                # event happened before first measurement: disregard it
                mmt_id = 0
                continue
            # write event data to selected measurement
            self.measurements[mmt_id].add_event_from_pandas(evt)
            
    def to_dict(self):
        d = {
            'diagnosis': self.diagnosis,
            'patient_id': self.patient_id,
            'measurements': [mmt.to_dict() for mmt in self.measurements]
        }
        return d

