#### Estimators - take care of data loading, model definition, training and predictions ####
#### Several convenience plotting functions are present as well ####
from losses import mae_last
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import models
from datasets import IOVar, LongitudinalOCTDataset
from sequences import load_sequences_from_pickle, split_sequences
from functools import partial
from copy import deepcopy
from datetime import datetime

def merged_config(config1, config2):
    """update config1 with config2"""
    res_config = deepcopy(config1)
    for key, value in res_config.items():
        res_config[key].update(config2.get(key, {}))
    return res_config

DEFAULT_CONFIG = {
    'sequence_data': {
        'sequence_fname': None,
        'load_sequences': False
    },
    'sequence_split': {
        'sequence_length': 5,
        'train_frac': 0.8,
        'val_frac': 0.1,
        'diagnosis': 'AMD',
        'seed': 42
    },
    'model': {
        'model_cls': models.SimpleLSTMModel,
        'input_vars': [IOVar.CUR_VA,IOVar.DELTA_T,IOVar.INJ_SHORT,IOVar.LENS_SURGERY],
        'output_vars': [IOVar.NEXT_VA,],
        'norm': [None, None, None, None, None],
        'sequence_length': 1,
        'num_dataloaders': 1,
    },
    'training': {
        'loss': tf.losses.mean_absolute_error,
        'metrics': [tf.losses.mean_absolute_error, mae_last],
        'tb_log_dir': None,
        'batch_size': 512,
        'epochs': 30,
    }
}

class Estimator():
    def __init__(self, config):
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = merged_config(DEFAULT_CONFIG, config)
        self.sequences_all = None
        self.sequences_split = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.model = None
        self.compiled_model = False
        self.callbacks = []
        
        
        # if possible, load sequences and create dataset from sequences
        self.sequences_all = None
        self.sequences_split = None
        if self.config['sequence_data']['load_sequences']:
            fname = self.config['sequence_data']['sequences_fname']
            self.log.info(f'Loading sequences from {fname}')
            # load sequences
            self.sequences_all = load_sequences_from_pickle(fname)
            # split sequences
            self.split_sequences()
            # create datasets 
            self.datasets_from_sequences()
            
        self.create_model()
        
            
    def split_sequences(self, sequences=None, return_values=False):
        assert (self.sequences_all is not None) or (sequences is not None), \
        "self.sequences_all is not loaded, and no sequences are provided"
        if sequences is not None:
            if self.sequences_all is not None:
                self.log.warn("Overwriting existing sequences_all with passed sequences")
            self.sequences_all = sequences
        
        config = self.config['sequence_split']
        sl = config.get('sequence_length')
        train_frac = config.get('train_frac')
        val_frac = config.get('val_frac')
        diagnosis = config.get('diagnosis')
        seed = config.get('seed', None)
        self.log.info(f'Splitting sequences of len {sl} into {train_frac} train, {val_frac} val (seed {seed})')
        train, val, test = split_sequences(self.sequences_all, sl, train_frac=train_frac, val_frac=val_frac, 
                                           diagnosis=diagnosis, seed=seed, log=self.log)
        if return_values:
            return train, val, test
        else:
            self.sequences_split = train, val, test
        
    def datasets_from_sequences(self, sequences=None, return_values=False):
        # create dataset from given sequences
        assert (self.sequences_split is not None) or (sequences is not None), \
        "self.sequences_split is None, and no sequences are provided"
        if sequences is not None:
            self.sequences_split = self.split_sequences(sequences, return_values=True)
        
        config = self.config['model']
        input_vars = config.get('input_vars')
        output_vars = config.get('output_vars')
        norm = config['norm']
        for i in range(len(norm)):
            if norm[i] == True:
                # should normalize, but no values are given
                # calculate normalization values
                io_var = (input_vars+output_vars)[i]
                self.log.info(f'Calculating normalization values for var {io_var}')
                norm_values = Estimator.get_mean_std(self.sequences_split[0], io_var)
                config['norm'][i] = norm_values
        datasets = [LongitudinalOCTDataset(seq, return_values=input_vars+output_vars, norm=norm, 
                                           sequence_length=config['sequence_length'], num_inputs=len(input_vars),
                                           num_parallel_calls=config['num_dataloaders']) for seq in self.sequences_split]
        if return_values:
            return datasets
        else:
            self.train_dataset = datasets[0]
            self.val_dataset = datasets[1]
            self.test_dataset = datasets[2]
                     
    def create_model(self):
        config = self.config['model']
        ModelClass = config['model_cls']
        self.log.info('Creating {} model with iputs {} and outputs {}'.format(ModelClass, config['input_vars'], config['output_vars']))
        model = ModelClass(config['input_vars'], config['output_vars'], config['sequence_length'])
        self.model = model
   
    def compile_model(self):
        if self.model is None:
            self.create_model()
        config = self.config['training']
        self.model.model.compile(
            optimizer = 'adam',
            loss = config['loss'],
            metrics = config['metrics']
        )
        # set up logging to tb
        callbacks = []
        if config['tb_log_dir'] is not None:
            logdir = "{}/{}".format(config['tb_log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
            file_writer = tf.summary.create_file_writer(logdir + "/metrics")
            file_writer.set_as_default()
            log_weights_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=partial(Estimator.log_variable_summaries, 
                                                                                      layers=self.model.log_layers))
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            self.callbacks.append(log_weights_callback)
            self.callbacks.append(tb_callback)
        self.compiled_model = True
    
    def train_model(self, verbose=0):
        if not self.compiled_model:
            self.compile_model()
        config = self.config['training']
        history = self.model.model.fit(
            x = self.train_dataset.dataset_for_training.batch(config['batch_size']),
            validation_data = self.val_dataset.dataset_for_training.batch(config['batch_size']),
            epochs = config['epochs'],
            verbose = verbose
        )
        # todo make this parameters
        return history
    
    def predict_model(self, dataset=None):
        if dataset is None:
            dataset = self.test
        config = self.config['training']
        result = self.model.model.predict(
            dataset.dataset_for_prediction.batch(config['batch_size']))
        return result
    
    def evaluate_model(self, dataset=None, verbose=0):
        if dataset is None:
            dataset = self.test
        config = self.config['training']
        result = self.model.model.evaluate(
            dataset.dataset_for_training.batch(config['batch_size']),
            verbose=verbose)
        return result
    
    @staticmethod        
    def log_variable_summaries(epoch, logs, layers):
        # variable summaries for tb logging
        def var_summary(name, data, step):
            tf.summary.histogram(name, data=data, step=step)
            tf.summary.scalar(name+'/mean', data=data.numpy().mean(), step=step)

        for layer in layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                units = int(layer.weights[0].shape[1]/4)
                name = layer.name
                for w, n in zip(layer.weights, ['W', 'U', 'b']):
                    for i, gate in enumerate(['input', 'forget', 'cellstate', 'output']):
                        if n == 'b':
                            current = w[i*units : (i+1)*units]
                        else: 
                            current = w[:, i*units : (i+1)*units]
                        var_summary('weights/{}/{}/{}'.format(name, n, gate), data=current, step=epoch)
            else:
                for w in layer.weights:
                    var_summary('weights/'+w.name, data=w, step=epoch)
        
    @staticmethod
    def get_mean_std(sequences, io_var, num_elem_to_use=None):
        """returns mean and std for sequences"""
        if io_var == IOVar.OCT:
            raise('Not implemented error')
        values = [io_var.get_data_from_mmt(mmt) for seq in sequences for mmt in seq.measurements[:-1]]
        mean = np.mean(values)
        std = np.std(values)
        return mean, std

    
def plot_train_curve(history, baseline_err, ylim=None):
    val = history.history['val_mae_last']
    train = history.history['mae_last']

    epochs = range(1, len(history.history['loss'])+1)
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.plot(epochs, val, label='val mae {:.3f}'.format(val[-1]))
    ax.plot(epochs, train, label='train mae {:.3f}'.format(train[-1]))
    ax.plot(epochs, [baseline_err for _ in epochs], label='baseline mae {:.3f}'.format(baseline_err))
    ax.set_title('mean absolute error of last prediction')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    
def plot_train_curve_comparison(histories, names, baseline_err, ylim=None):
    vals = [h.history['val_mae_last'] for h in histories]
    trains = [h.history['mae_last'] for h in histories]
    
    epochs = range(1, len(histories[0].history['loss'])+1)
    fig, axes = plt.subplots(1,2, figsize=(15,5))
    for i in range(len(histories)):
        axes[0].plot(epochs, vals[i], label=names[i] + ' mae: {:.3f}'.format(vals[i][-1]))
        axes[1].plot(epochs, trains[i], label=names[i] + ' mae: {:.3f}'.format(trains[i][-1]))
    axes[0].set_title('Val mean absolute error of last prediction')
    axes[1].set_title('Train mean absolute error of last prediction')
    for ax in axes:
        ax.plot(epochs, [baseline_err for _ in epochs], label='baseline mae {:.3f}'.format(baseline_err))
        if ylim is not None:
            ax.set_ylim(*ylim)