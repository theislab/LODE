#### model definitions ####
import tensorflow as tf
from datasets import IOVar

class SimpleLSTMModel():
    def __init__(self, input_vars, output_vars, sequence_length):
        """
        Basic LSTM architecture which concatenates all inputs and predicts a single output
        """
        assert IOVar.OCT not in input_vars, "cannot use input OCT in SimpleLSTM"
        assert len(output_vars) == 1, "only support one output with this architecture!"
        self.input_vars = input_vars
        self.sequence_length = sequence_length
        self.log_layers = []
        
        self.inputs = [tf.keras.layers.Input(io_var.get_shape(sequence_length=sequence_length)[1:]) for io_var in self.input_vars]
        X = self._input_layers(self.inputs)
        X = self._embedding_layers(X)
        X = self._lstm_layers(X)
        X = self._output_layers(X)
        self.outputs = [X]
        
        self.model = tf.keras.Model(self.inputs, self.outputs)
        self.layers = self.model.layers
        self.summary = self.model.summary

    def _input_layers(self, X):
        # concatenate inputs, then split along sequence dimension
        if len(X) > 1:
            X = tf.keras.layers.concatenate(X, axis=-1)
        else:
            X = X[0]
        return X
    
    def _embedding_layers(self, X):
        X = tf.keras.layers.Lambda(lambda x: tf.split(x, self.sequence_length, axis=1))(X)
        dense_layer = tf.keras.layers.Dense(32, activation='relu')
        X = [dense_layer(x) for x in X]
        if len(X) > 1:
            X = tf.keras.layers.concatenate(X, axis=1)
        else:
            X = X[0]
        
        self.log_layers.append(dense_layer)
        return X
    
    def _lstm_layers(self, X):
        lstm_layer = tf.keras.layers.LSTM(units=64, return_sequences=True)
        X = lstm_layer(X)
        
        self.log_layers.append(lstm_layer)
        return X
    
    def _output_layers(self, X):
        # split X and calculate output for each element
        X = tf.keras.layers.Lambda(lambda x: tf.split(x, self.sequence_length, axis=1))(X)
        output_layer = tf.keras.layers.Dense(1, activation=None)
        X = [output_layer(x) for x in X]
        if len(X) > 1:
            X = tf.keras.layers.concatenate(X, axis=1)
        else:
            X = X[0]
        self.log_layers.append(output_layer)
        return X
