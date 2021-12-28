import logging 
from enum import Enum
import tensorflow as tf
import numpy as np

def bin_diffva(diff_va, measurement_error=0.15):
    if diff_va < -measurement_error:
        return [1,0,0]  # improve
    elif diff_va > measurement_error:
        return [0,1,0]  # worse
    else:
        return [0,0,1]  # same
    
class IOVar(Enum):
    CUR_VA = 'cur_va'
    NEXT_VA = 'next_va'
    OCT = 'oct'
    DELTA_T = 'delta_t'
    INJ_SHORT = 'inj_short'
    INJ_LONG = 'inj_long'
    LENS_SURGERY = 'lens_surgery'
    ALL_FEATURES = 'all_features'
    CHECKUP1_CUR_VA = 'checkup1_cur_va'
    CHECKUP1_DIFF_VA_CLASS = 'checkup1_diff_va_class'
    CHECKUP1_ALL_FEATURES = 'checkup1_all_features'
    CHECKUP2_CUR_VA = 'checkup2_cur_va'
    CHECKUP2_DIFF_VA_CLASS = 'checkup2_diff_va_class'
    CHECKUP2_ALL_FEATURES = 'checkup2_all_features'
    
    def get_data_from_mmt(self, mmt):
        # self is the member
        cls = self.__class__
        if self == cls.CUR_VA:
            val = getattr(mmt, 'cur_va', 0)
            # round to two digits
            return np.round(val, decimals=2)
        elif self == cls.NEXT_VA:
            val = getattr(mmt, 'next_va', 0)
            return np.round(val, decimals=2)
        elif self == cls.OCT:
            return getattr(mmt, 'oct_path', "")    
        elif self == cls.DELTA_T:
            return getattr(mmt, 'delta_t', 0)
        elif self == cls.INJ_SHORT:
            return sum(getattr(mmt, 'injections', [0,0,0,0,0,0,0,0]))
        elif self == cls.INJ_LONG:
            return getattr(mmt, 'injections', [0,0,0,0,0,0,0,0])
        elif self == cls.LENS_SURGERY:
            return getattr(mmt, 'lens_surgery', 0)
        elif self == cls.ALL_FEATURES:
            return getattr(mmt, 'features', [0 for _ in range(12)])
        elif 'checkup1' in self.value or 'checkup2' in self.value:
            # checkup IOVar: get checkup
            checkup_name = self.value.split('_')[0]
            checkup_mmt = getattr(mmt, checkup_name, None)
            if 'diff_va_class' in self.value:
                if checkup_mmt is None:
                    return [0,0,0]
                else:
                    return bin_diffva(checkup_mmt.cur_va - mmt.cur_va)
            else:
                # get data from checkup_mmt by creating new io_var with the desired value 
                # and using its get_data_from_mmt function
                io_var = IOVar('_'.join(self.value.split('_')[1:]))
                return io_var.get_data_from_mmt(checkup_mmt)

        
    def get_dtype(self, flavour='tf'):
        cls = self.__class__
        if 'oct' in self.value:
            return tf.string if flavour == 'tf' else np.string
        else:
            return tf.float32 if flavour == 'tf' else np.float32

        
    def get_shape(self, num_samples=None, sequence_length=None):
        cls = self.__class__
        if 'inj_long' in self.value:
            return (num_samples, sequence_length, 8)
        elif 'all_features' in self.value:
            return (num_samples, sequence_length, 12)
        elif 'diff_va_class' in self.value:
            return (num_samples, sequence_length, 3)
        else:
            return (num_samples, sequence_length, 1)
        
        
class LongitudinalOCTDataset():
    def __init__(self, sequences, return_values=(IOVar.OCT,IOVar.NEXT_VA), norm=None, sequence_length=None, num_inputs=-1, 
                 num_parallel_calls=1, oversample=False):
        """
        Create tf.Dataset from sequences. The dataset will return return_values assembled from sequences with shape
        ((num_samples, sequence_length, var1_shape),...) 
        Sequence loading and splitting is done outside of this class.
        NOTE: does not support nested tuples for return_values
        Args:
            sequences: list of MeasurementSequences
            return_values: tuple, definition of variables that the dataset returns
            norm: list of tuples (mean, std). Values for normalizing each entry X of return_values X: (X-mean)/std. 
                Can be None for individual entries. Shape should fit return_values
            sequence_length: if None, take length of first sequence as sequence length for dataset.
                Pad with zeros, if sequence is shorter than sequence_length
            oversample: (optional, only for classification targets), if true, balances samples according to their classes.
                Epochs will be num_classes*max_samples_per_class long.
                Useful for imbalanced classification targets.
        """
        self.log = logging.getLogger(self.__class__.__name__)
        self.sequences = sequences
        self.return_values = return_values
        if norm is None:
            norm = tuple([None for _ in return_values])
        self.norm = norm
        self.oversample = oversample
        if sequence_length is None:
            sequence_length = len(self.sequences[0])
        self.sequence_length = sequence_length
        self.num_samples = len(self.sequences)
        self.num_parallel_calls = num_parallel_calls
        self.num_inputs = num_inputs
        
        self.dataset = self._get_dataset()
        self.dataset_for_training = self._get_dataset_for_training()
        self.dataset_for_prediction = self._get_dataset_for_prediction()
    
    def _get_dataset(self):
        data = self._prepare_data()
        def generator():
            for i in range(self.num_samples):
                yield tuple([el[i] for el in data])
        
        output_types = tuple([io_var.get_dtype('tf') for io_var in self.return_values])
        output_shapes = tuple([io_var.get_shape(sequence_length=self.sequence_length)[1:] for io_var in self.return_values])
        dataset = tf.data.Dataset.from_generator(generator=generator, 
                                                 output_types=output_types, 
                                                 output_shapes=output_shapes)
        #dataset = dataset.shuffle(self.num_samples, seed=42, reshuffle_each_iteration=False)
        
        # map octs
        if IOVar.OCT in self.return_values:
            idx = self.return_values.index(IOVar.OCT)
            def read_oct(*data):
                # read OCT
                data[idx] = 'loaded OCT' # TODO
                return data
            dataset = dataset.map(read_oct, num_parallel_calls=self.num_parallel_calls)
            
        # normalize values
        def norm(*data):
            normed_data = []
            for i in range(len(data)):
                if self.norm[i] is not None:
                    mean, std = self.norm[i]
                    normed_data.append((data[i]-mean)/std)
                else:
                    normed_data.append(data[i])
            return tuple(normed_data)
        dataset = dataset.map(norm, num_parallel_calls=self.num_parallel_calls)
        return dataset
    
    def _get_dataset_for_training(self):
        def map_fn(*data):
            return data[:self.num_inputs], data[self.num_inputs:]
        dataset = self.dataset.map(map_fn)
        
        if self.oversample:
            per_class_datasets = self.get_per_class_datasets(dataset)
            if per_class_datasets is None:
                self.log.warn('should oversample but could not calculate per_class_datasets')
            else:
                self.log.info('oversampling data')
                def len_ds(ds):
                    return ds.reduce(0, lambda x,_: x+1)
                len_datasets = np.array([len_ds(ds) for ds in per_class_datasets])
                # make all datasets repeat
                per_class_datasets = [ds.repeat() for ds in per_class_datasets]
                # combine datasets and take n*max_len samples from it
                dataset = tf.data.experimental.sample_from_datasets(per_class_datasets).take(len_datasets.max()*len(per_class_datasets))
                
        return dataset

    def _get_dataset_for_prediction(self):
        def map_fn(*data):
            return [data[:self.num_inputs]]
        return self.dataset.map(map_fn)
     
    def _prepare_data(self):
        """
        Read data from self.sequences according to self.return_values. 
        Replace oct image with oct_path -- this will be loaded later using the generator
        """
        data = []
        for io_var in self.return_values:
            # get number of features for each config element
            arr = np.zeros(io_var.get_shape(self.num_samples, self.sequence_length), dtype=io_var.get_dtype("np"))
            data.append(arr)
        
        for i, seq in enumerate(self.sequences):
            for j, mmt in enumerate(seq.measurements[::-1]):
                # go through measurements in reverse order to ensure we get the latest ones
                k = self.sequence_length-j-1
                if k < 0:
                    break
                for l, io_var in enumerate(self.return_values):
                    data[l][i,k] = io_var.get_data_from_mmt(mmt)
        return data           
        
    def get_value_list(self, values=[], batch_size=512):
        idx = [self.return_values.index(io_var) for io_var in values]
        res = [np.zeros(io_var.get_shape(0, self.sequence_length)) for io_var in values]
        for X in self.dataset.batch(batch_size):
            for i in range(len(res)):
                res[i] = np.concatenate([res[i], X[idx[i]]])
        return res
    
    def get_per_class_datasets(self, dataset):
        # get one dataset per class for classes defined in the LAST return value
        # class is determined by the LAST sequence
        if self.return_values[-1].value in (IOVar.CHECKUP1_DIFF_VA_CLASS.value, IOVar.CHECKUP2_DIFF_VA_CLASS.value):
            num_classes = list(self.dataset.take(1))[0][-1].shape[-1]
            per_class_datasets = []
            for cl in range(num_classes):
                per_class_datasets.append(dataset.filter(lambda features, label: tf.argmax(label, axis=-1)[-1,-1]==cl))
            return per_class_datasets
        else:
            return None
    