#### Estimators - take care of data loading, model definition, training and predictions ####
#### Several convenience plotting functions are present as well ####
from losses import mae_last
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import models
from datasets import IOVar, LongitudinalOCTDataset
from sequences import load_sequences_from_pickle, split_sequences, Measurement
from functools import partial
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from utils import merged_config

DEFAULT_CONFIG = {
    'sequence_data': {
        'sequence_fname': None,
        'load_sequences': False,
        'features_fname': None,
        'num_checkups': 1,
    },
    'sequence_split': {
        'sequence_length': 5, # instead of one length, can also define 'min_len' and 'max_len'
        'train_frac': 0.8,
        'val_frac': 0.1,
        'diagnosis': 'AMD',
        'seed': 42
    },
    'model': {
        'model_cls': models.SimpleLSTMModel,
        'model_kwargs': {},
        'input_vars': [IOVar.CUR_VA,IOVar.DELTA_T,IOVar.INJ_SHORT,IOVar.LENS_SURGERY],
        'output_vars': [IOVar.NEXT_VA,],
        'norm': [None, None, None, None, None],
        'sequence_length': 1,
        'num_dataloaders': 1,
        'oversample_classification': False,
    },
    'training': {
        'loss': tf.losses.mean_absolute_error,
        'metrics': [tf.losses.mean_absolute_error, mae_last],
        'tb_log_dir': None,
        'batch_size': 512,
        'epochs': 30,
        'learning_rate': 0.001,
    }
}

class Estimator():
    def __init__(self, config):
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = merged_config(DEFAULT_CONFIG, config)
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
            # load sequences and prepare (e.g. remove checkups)
            self.sequences_all = self._prepare_sequences(load_sequences_from_pickle(fname))
            # split sequences
            self.sequences_split = self.split_sequences()
            # create datasets
            self.train_dataset, self.val_dataset, self.test_dataset = self.datasets_from_sequences()
            
        self.create_model()
        
            
    def _prepare_sequences(self, sequences, grouped_features=None):
        """prepare sequences for splitting and training.
        If necessary, remove checkup_sequences and add to last measurement.
        Add features to sequences
        """
        config = self.config['sequence_data']
        # remove checkups
        num_checkups = config.get('num_checkups')
        if num_checkups > 0:
            self.log.info(f"Removing {num_checkups} checkups from sequences")
            res_sequences = []
            mmt_id = -num_checkups-1
            for seq in sequences:
                seq = deepcopy(seq)
                for i in range(num_checkups):
                    checkup_id = -(num_checkups-i)
                    checkup_no = i+1
                    # add to last mmt
                    setattr(seq.measurements[mmt_id], f'checkup{checkup_no}', deepcopy(seq.measurements[checkup_id]))
                for i in range(num_checkups):
                    # remove last num_checkups measurements
                    seq.remove_measurement(len(seq)-1, event_assignment='next')  # TODO event assignment could be a parameter!
                res_sequences.append(seq)
        else:
            # TODO assume that need to remove last measurement, because contains target VA value 
            # (which is saved in next_va in previous mmt)
            res_sequences = sequences
        # load features 
        res_sequences = self._load_features_for_sequences(res_sequences, grouped_features)
        return res_sequences
    
    def _load_features_for_sequences(self, res_sequences, grouped_features=None):
        """loads segmented clinical features and matches features to each sequence"""
        features_fname = self.config['sequence_data']['features_fname']
        if features_fname is not None and grouped_features is None:
            # load features from file
            self.log.info('Loading features from {}'.format(features_fname))
            features = pd.read_csv(features_fname, index_col=0)
            features.columns = Measurement.FEATURES + ['dicom_name','frame','patient_id','oct_path','laterality','study_date']
            # summarize features over all frames of each oct
            grouped_features = features.groupby(['oct_path'])[Measurement.FEATURES].sum()
        if grouped_features is None:
            # no features to be added
            return res_sequences
        # add features
        self.log.info(f'Adding features to sequences')
        for seq in tqdm(res_sequences):
            seq.add_features_from_pandas(grouped_features, normalize='retina')
        return res_sequences
    
    
    def split_sequences(self, sequences=None, grouped_features=None):
        assert (self.sequences_all is not None) or (sequences is not None), \
        "self.sequences_all is not loaded, and no sequences are provided"
        if sequences is not None:
            if self.sequences_all is not None:
                self.log.warn("Overwriting existing sequences_all with passed sequences")
            self.sequences_all = self._prepare_sequences(sequences, grouped_features)
        
        config = self.config['sequence_split']
        min_len = config.get('min_len', config.get('sequence_length', None))
        max_len = config.get('max_len', config.get('sequence_length', None))
        train_frac = config.get('train_frac')
        val_frac = config.get('val_frac')
        diagnosis = config.get('diagnosis')
        seed = config.get('seed', None)
        only_hard_sequences = config.get('only_hard_sequences', False)
        self.log.info(f'Splitting sequences of len {min_len}-{max_len} into {train_frac} train, {val_frac} val (seed {seed})')
        train, val, test = split_sequences(self.sequences_all, min_len=min_len, max_len=max_len, 
                                           train_frac=train_frac, val_frac=val_frac, 
                                           diagnosis=diagnosis, seed=seed, log=self.log, only_hard_sequences=only_hard_sequences)
        return train, val, test


    def datasets_from_sequences(self, sequences=None, grouped_features=None):
        # create dataset from given sequences
        assert (self.sequences_split is not None) or (sequences is not None), \
        "self.sequences_split is None, and no sequences are provided"
        if sequences is not None:
            self.sequences_split = self.split_sequences(sequences, grouped_features)
        
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
        print('resulting norm values', norm)
        # TODO use below code for implementing class weighting
        #class_weights = False
        #if config['class_weighting']:
        #    assert output_vars[0] in (IOVar.CHECKUP1_DIFF_VA_CLASS, IOVar.CHECKUP2_DIFF_VA_CLASS), \
        #        'class weighting specified, but no classification target!'
        #    class_weights = Estimator.get_class_weights(self.sequences_split[0], io_var=output_vars[0])
        #    self.log.info(f'Calculated class weights: {oversample_weights}')

        datasets = [LongitudinalOCTDataset(seq, return_values=input_vars+output_vars, norm=norm,
                                           oversample=config['oversample_classification'],
                                           sequence_length=config['sequence_length'], num_inputs=len(input_vars),
                                           num_parallel_calls=config['num_dataloaders']) for seq in self.sequences_split]
        return datasets


    
                
    def create_model(self):
        config = self.config['model']
        ModelClass = config['model_cls']
        self.log.info('Creating {} model with iputs {} and outputs {}'.format(ModelClass, config['input_vars'], config['output_vars']))
        model = ModelClass(config['input_vars'], config['output_vars'], config['sequence_length'], **config['model_kwargs'])
        self.model = model

    def compile_model(self):
        if self.model is None:
            self.create_model()
        config = self.config['training']
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        self.model.model.compile(
            optimizer = optimizer,
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
            dataset = self.val_dataset
        config = self.config['training']
        result = self.model.model.predict(
            dataset.dataset_for_prediction.batch(config['batch_size']))
        return result
    
    def evaluate_model(self, dataset=None, verbose=0):
        if dataset is None:
            dataset = self.val_dataset
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
    def get_mean_std(sequences, io_var):
        """returns mean and std for sequences"""
        if io_var == IOVar.OCT:
            raise('Not implemented error')
        values = [io_var.get_data_from_mmt(mmt) for seq in sequences for mmt in seq.measurements]
        if 'all_features' in io_var.value:
            # calculate quantiles of data
            values = np.array(values)
            q01, q09 = np.quantile(values, (0.1, 0.9), axis=0)
            for i in range(values.shape[-1]):
                values[:,i][values[:,i] < q01[i]] = q01[i]
                values[:,i][values[:,i] > q09[i]] = q09[i]
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        return mean, std
    
    @staticmethod
    def get_class_weights(sequences, io_var):
        # TODO currently not used, implemented for oversamplling but did not need to use!
        labels = []
        for seq in sequences:
            labels.append(np.argmax(io_var.get_data_from_mmt(seq.measurements[-1]), axis=-1))
        labels = np.array(labels)
        num_per_class = np.zeros(labels.max()+1)
        for cl in range(num_per_class.shape[0]):
            num_per_class[cl] = (labels==cl).sum()
        class_weights = 1 / num_per_class
        class_weights = class_weights / np.sum(class_weights)
        return class_weights
    
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
        ax.legend()
        
def plot_history(history, baseline_err=None, ylim=None, keys=['mae_last']):
    """
    Compare train and val progress of training history.
    Plots one plot per measure (keys in history dict). 
    Per default plots train mae and val mae
    Args:
        keys: list of keys to be plotted. For each entry in keys, a new plot is made. 
            Each entry in keys may be a list of keys to be plotted in that subplot or a single string
    """
    if not isinstance(baseline_err, list):
        baseline_err = [baseline_err for _ in keys]
    epochs = range(1, len(history.history['loss'])+1)
    num_cols = min(3, len(keys))
    num_rows = int(np.ceil(len(keys)/3))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols,5*num_rows), squeeze=False)
    # plot train and val for all keys
    for i, (key, ax) in enumerate(zip(keys, axes.flat)):
        if not isinstance(key, list):
            key = [key]
        for k in key:
            ax.plot(epochs, history.history[k], label='{} {:.3f}'.format(k, history.history[k][-1]))
            ax.plot(epochs, history.history['val_'+k], label='val {} {:.3f}'.format(k, history.history['val_'+k][-1]))
        if baseline_err[i] is not None:
            ax.plot(epochs, [baseline_err[i] for _ in epochs], label='baseline mae {:.3f}'.format(baseline_err))
        ax.set_title(key)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()

# TODO these functions should replace the above!
def plot_history_comparison(histories, names, baseline_err=None, ylim=None, keys=['mae_last']):
    """
    Compare training histories of different models.
    Plots a grid with columns showing train and val values, and rows showing different measures (keys in history dict)
    Per default compares mae of histories
    Args:
        keys: list of keys to be plotted. For each entry in keys, a new plot is made. 
            Each entry in keys may be a list of keys to be plotted in that subplot or a single string
    """
    if not isinstance(baseline_err, list):
        baseline_err = [baseline_err for _ in keys]
    epochs = range(1, len(histories[0].history['loss'])+1)
    # set up plot
    num_cols = 2
    num_rows = len(keys)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols,5*num_rows), squeeze=False)
    for i, row in enumerate(axes):
        key = keys[i]
        if not isinstance(key, list):
            key = [key]
        for k in key:
            for name, history in zip(names, histories):
                row[0].plot(epochs, history.history[k], label='{} {}: {:.3f}'.format(name, k, history.history[k][-1]))
                row[1].plot(epochs, history.history['val_'+k], label='{} {} {:.3f}'.format(name, k, history.history['val_'+k][-1]))
        row[0].set_title('train {}'.format(key))
        row[1].set_title('val {}'.format(key))
        for ax in row:
            if baseline_err[i] is not None:
                ax.plot(epochs, [baseline_err for _ in epochs], label='baseline mae {:.3f}'.format(baseline_err))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.legend()