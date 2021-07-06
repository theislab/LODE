#### model definitions ####
import tensorflow as tf
from datasets import IOVar
from utils import merged_config
import json
import logging

def makelist(val, len_list=1):
    if isinstance(val, list):
        assert len(val) == len_list, "val is list, but has length {} instead of {}".format(len(val), len_list)
        return val
    return [val for _ in range(len_list)]
    

class SequenceModel():
    def __init__(self, input_vars, output_vars, sequence_length):
        """
        Superclass for sequences networks with embedding - sequence modelling - output architecture.
        Contains functions for creating groups of layers.
        Specific model architecture should be created by implementing subclasses in __init__ functions.
        """
        self.log = logging.getLogger(self.__class__.__name__)
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.sequence_length = sequence_length
        # list of layers that is logged to tensorboard
        self.log_layers = []
        
    def _concatenate_inputs(self, X):
        """
        assumes a list of inputs X (e.g. va, delta_t, features, ...)
        concatenates inputs
        """
        if len(X) > 1:
            X = tensorflow.keras.layers.concatenate(X, axis=-1)
        else:
            X = X[0]
        return X
    
    def _flatten(self, X):
        """
        assumes an input X with shape (batch, sequence, features).
        flattens sequence and features dimensions of X, returning X with shape (batch, _)
        """
        X = tensorflow.keras.layers.Flatten()(X)
        return X
    
    def _dense_shared_layers(self, X, num_layers=1, units=32, activation='relu', log=False):
        """
        creates `num_layers` dense layers with `units` units. 
        Input X is split in individual measurements (axis 1) and the same dense 
        layer is applied to all measurements (weight sharing)
        Args:
            num_layers: number of dense layers
            units: int or list of ints, number of units in the embedding layers
            activation: string, activation function
            log: log weights to tensorboard
        """
        if num_layers == 0:
            return X
        X = tensorflow.keras.layers.Lambda(lambda x: tf.split(x, self.sequence_length, axis=1))(X)
        units = makelist(units, num_layers)
        for i in range(num_layers):
            dense_layer = tf.tensorflow.keras.layers.Dense(units[i], activation=activation)
            X = [dense_layer(x) for x in X]
            if log:
                self.log_layers.append(dense_layer)
        if len(X) > 1:
            X = tf.tensorflow.keras.layers.concatenate(X, axis=1)
        else:
            X = X[0]
        return X
    
    def _lstm_layers(self, X, num_layers=1, units=64, log=False):
        """
        creates `num_layers` lstm layers with `units` units. Process the entire sequences at once
        Args:
            num_layers: number of embedding layers
            units: int or list of ints, number of units in the lstm layers
            log: log weights to tensorboard
        """
        if num_layers == 0:
            return X
        units = makelist(units, num_layers)
        for i in range(num_layers):
            lstm_layer = tf.tensorflow.keras.layers.LSTM(units=units[i], return_sequences=True)
            X = lstm_layer(X)
            if log:
                self.log_layers.append(lstm_layer)
        return X
    
    def _dense_layers(self, X, num_layers=1, units=64, activation='relu', log=False):
        """
        creates `num_layers` dense layers with `units` units. Process the entire sequence at once
        Args:
            num_layers: number of embedding layers
            units: int or list of ints, number of units in the lstm layers
            activation: string, activation function
            log: log weights to tensorboard
        """
        if num_layers == 0:
            return X
        units = makelist(units, num_layers)
        for i in range(num_layers):
            dense_layer = tf.tensorflow.keras.layers.Dense(units[i], activation=activation)
            X = dense_layer(X)
            if log:
                self.log_layers.append(dense_layer)
        return X
    
    def build_model(self):
        """create tensorflow.keras model from self.inputs and self.outputs"""
        self.model = tf.tensorflow.keras.Model(self.inputs, self.outputs)
        self.layers = self.model.layers
        self.summary = self.model.summary

class SimpleANNModel(SequenceModel):
    
    config = {
        'encoder_layers': {
            'num_layers': 1,
            'units': 32,
            'activation': 'relu',
            'log': False
        },
        'dense_layers':{
            'num_layers': 1,
            'units': 64,
            'activation': 'relu',
            'log': False
        },
        'decoder_layers': {
            'num_layers': 0,
            'units': 32,
            'activation': 'relu',
            'log': False
        },
        'output_layer':{
            'units': 1,
            'activation': None,
            'log': False,
        }
    }
    
    def __init__(self, input_vars, output_vars, sequence_length, **kwargs):
        """
        Basic ANN architecture which concatenates all inputs and predicts one output for the entire sequence
        """
        assert IOVar.OCT not in input_vars, "cannot use input OCT in SimpleLSTM"
        assert len(output_vars) == 1, "only support one output with this architecture!"
        super().__init__(input_vars, output_vars, sequence_length)
        self.config = merged_config(self.config, kwargs)
        self.log.info(json.dumps(self.config, indent=4))
        
        # create model architecture
        self.inputs = [tf.tensorflow.keras.layers.Input(io_var.get_shape(sequence_length=sequence_length)[1:]) for io_var in self.input_vars]
        X = self._concatenate_inputs(self.inputs)
        X = self._dense_shared_layers(X, **self.config['encoder_layers'])
        X = self._flatten(X)
        X = self._dense_layers(X, **self.config['dense_layers'])
        X = self._dense_layers(X, **self.config['decoder_layers'])
        X = self._dense_layers(X, num_layers=1, **self.config['output_layer'])
        # reshape to output results of shape (batch, 1, output_units) 
        # - needed for easier metrics calculation and consistency with LSTM model
        X = tf.tensorflow.keras.layers.Reshape((1,self.config['output_layer']['units']))(X)
        self.outputs = [X]
        
        self.build_model()



class SimpleLSTMModel(SequenceModel):
    
    config = {
        'encoder_layers': {
            'num_layers': 1,
            'units': 32,
            'activation': 'relu',
            'log': False
        },
        'lstm_layers':{
            'num_layers': 1,
            'units': 64,
            'log': False
        },
        'decoder_layers': {
            'num_layers': 0,
            'units': 32,
            'activation': 'relu',
            'log': False
        },
        'output_layer':{
            'units': 1,
            'activation': None,
            'log': False,
        }
    }
    
    def __init__(self, input_vars, output_vars, sequence_length, **kwargs):
        """
        Basic LSTM architecture which concatenates all inputs and predicts a single output for each measurement
        """
        assert IOVar.OCT not in input_vars, "cannot use input OCT in SimpleLSTM"
        assert len(output_vars) == 1, "only support one output with this architecture!"
        super().__init__(input_vars, output_vars, sequence_length)
        self.__class__.config.update(kwargs)
        self.config = merged_config(self.config, kwargs)
        self.log.info(json.dumps(self.config, indent=4))
        
        # create model architecture
        self.inputs = [tf.tensorflow.keras.layers.Input(io_var.get_shape(sequence_length=sequence_length)[1:]) for io_var in self.input_vars]
        X = self._concatenate_inputs(self.inputs)
        X = self._dense_shared_layers(X, **self.config['encoder_layers'])
        X = self._lstm_layers(X, **self.config['lstm_layers'])
        X = self._dense_shared_layers(X, **self.config['decoder_layers'])
        X = self._dense_shared_layers(X, num_layers=1, **self.config['output_layer'])
        self.outputs = [X]
        
        self.build_model()
