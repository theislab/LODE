import logging 
from enum import Enum
import tensorflow as tf
import numpy as np

class IOVar(Enum):
    CUR_VA = 1
    NEXT_VA = 2
    OCT = 3
    DELTA_T = 4
    INJ_SHORT = 5
    INJ_LONG = 6
    LENS_SURGERY = 7
    
    def get_data_from_mmt(self, mmt):
        # self is the member
        cls = self.__class__
        if self == cls.CUR_VA:
            return mmt.cur_va
        elif self == cls.NEXT_VA:
            return mmt.next_va
        elif self == cls.OCT:
            return mmt.oct_path
        elif self == cls.DELTA_T:
            return mmt.delta_t
        elif self == cls.INJ_SHORT:
            return sum(mmt.injections)
        elif self == cls.INJ_LONG:
            return mmt.injections
        elif self == cls.LENS_SURGERY:
            return int(mmt.lens_surgery)
        
    def get_dtype(self, flavour='tf'):
        cls = self.__class__
        if self == cls.OCT:
            return tf.string if flavour == 'tf' else np.string
        else:
            return tf.float32 if flavour == 'tf' else np.float32

        
    def get_shape(self, num_samples=None, sequence_length=None):
        cls = self.__class__
        if self == cls.INJ_LONG:
            return (num_samples, sequence_length, 8)
        else:
            return (num_samples, sequence_length, 1)
        
        
class LongitudinalOCTDataset():
    def __init__(self, sequences, return_values=(IOVar.OCT,IOVar.NEXT_VA), norm=None, sequence_length=None, num_inputs=-1, 
                 num_parallel_calls=1):
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
            sequence_length: if None, take length of first sequence - 1 as sequence length for dataset.
                Pad with zeros, if sequence is shorter than sequence_length
        """
        self.log = logging.getLogger(self.__class__.__name__)
        self.sequences = sequences
        self.return_values = return_values
        if norm is None:
            norm = tuple([None for _ in return_values])
        self.norm = norm
        if sequence_length is None:
            sequence_length = len(self.sequences[0]) - 1
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
        return self.dataset.map(map_fn)

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
            for j, mmt in enumerate(seq.measurements[::-1][1:]):
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
    