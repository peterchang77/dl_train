import os, glob, pickle
import numpy as np, pandas as pd
from dl_utils import io
from dl_utils.db import DB, find_matching_files, FUNCS
from dl_utils.general import *

class Client():

    def __init__(self, *args, **kwargs):
        """
        Method to set default variables and paths

        :params

          (str) DS_PATH : path to dataset root 
          (str) DS_FILE : path to summary CSV file

        """
        # --- Parse args
        kwargs = self.parse_args(*args, **kwargs)
        self.DS_PATH = kwargs['DS_PATH']
        self.DS_FILE = kwargs['DS_FILE']

        self.df = None

        self.cohorts = {'train': {}, 'valid': {}}
        self.indices = {'train': {}, 'valid': {}}
        self.current = {'train': {}, 'valid': {}}

        self.apply = None
        self.sampling_rates = None 
        self.set_training_rates()

        if os.path.exists(self.DS_FILE):
            self.load_csv()

    def parse_args(self, *args, **kwargs):
        """
        Method to parse arguments into final kwargs dict 

        """
        DEFAULTS = {
            'DS_PATH': '',
            'DS_FILE': ''}

        env = {k: os.environ[k] for k in DEFAULTS if k in os.environ}

        # --- Convert args to args_ dict 
        args_ = {}
        for arg in args:

            if type(arg) is str:

                if os.path.isdir(arg):
                    args_['DS_PATH'] = arg

                ext = arg.split('.')[-1]
                if ext in ['csv', 'gz']:
                    args_['DS_FILE'] = arg

        return {**DEFAULTS, **env, **args_, **kwargs}

    def check_data_is_loaded(func):
        """
        Method (decorator) to ensure the self.data / self.meta is loaded

        """
        def wrapper(self, *args, **kwargs):

            if self.df is None:
                self.load_csv()

            return func(self, *args, **kwargs)

        return wrapper

    def load_csv(self, DS_FILE=None):

        DS_FILE = DS_FILE or self.DS_FILE

        if os.path.exists(DS_FILE):
            self.df = pd.read_csv(DS_FILE)

    def to_csv(self, DS_FILE=None):

        DS_FILE = DS_FILE or self.DS_FILE
        os.makedirs(os.path.dirname(DS_FILE), exist_ok=True)
        self.df.to_csv(DS_FILE)

    def init_funcs(self, funcs_def=None, **kwargs):
        """
        :params

          (list) funcs  : list of function names from dl_utils.db.funcs

        """
        if funcs_def is None:

            funcs_def = [{

                'func': 'coord',
                'kwargs': {'lbl': 'lbl'}}, {

                'func': 'stats',
                'kwargs': {'dat': 'dat'}}, {

                'func': 'label',
                'kwargs': {'lbl': 'lbl', 'classes': kwargs.get('classes', 2)}

            }]

        # -- Parse into funcs, kwargs
        self.apply = {
            'funcs': [FUNCS[d['func']] for d in funcs_def],
            'kwargs': [d['kwargs'] for d in funcs_def]}

        # --- Add load function
        self.apply['load'] = io.load

    def make_summary(self, query=None, db=None, join=None, N_FOLDS=5, DS_FILE=None):
        """
        Method to read all data and make summary CSV file 

        :return

          (dict) summary : {

            'data': [{paths_00}, {paths_01}, ...],
            'meta': {
              'index': [0, 0, 0, ..., 1, 1, 1, ...],
              'coord': [0, 1, 2, ..., 0, 1, 2, ...],
              'mu': [...],                              # mu of each individual dat volume
              'sd': [...],                              # sd of each individual dat volume
              0: [1, 1, 0, 0, 1, ...],                  # presence of class == 0 at slice pos
              1: [1, 1, 0, 0, 1, ...],                  # presence of class == 1 at slice pos
              ...
            } 
          }

        IMPORTANT: this is a default template; please modify as needed for your data

        """
        # --- Perform query
        if db is None:
            query = {**{'root': self.DS_PATH}, **query}
            db = DB(configs={'query': query})

        # --- Apply 
        if self.apply is None:
            self.init_funcs()

        df = db.apply(**self.apply)

        # --- Join fnames + valid
        if join is None:
            join = db.fnames.columns.tolist() + ['valid']
            series = db.fnames
        else:
            join.append('valid')
            series = db.df_merge(rename=False)

        valids = np.arange(series.shape[0]) % N_FOLDS
        series['valid'] = valids[np.random.permutation(valids.size)]

        cols = join + df.columns.tolist()
        df = df.join(series[join])
        df = df[cols]

        # --- Serialize
        self.df = df
        self.to_csv(DS_FILE)

        # --- Final output
        printd('Summary complete: %i patients | %i slices' % (series.shape[0], df.shape[0]))

    def load(self, data, **kwargs):

        if type(data) is not str:
            return data

        LOAD_FUNC = {
            'hdf5': self.load_hdf5}

        ext = data.split('.')[-1]

        if ext in LOAD_FUNC:
            return LOAD_FUNC[ext](data, **kwargs)

        else:
            printd('ERROR provided extension is not supported: %s' % ext)
            return None, {} 

    def load_dict(self, data, **kwargs):

        assert type(data) is dict

        arrays = {}
        for key, val in data.items():
            arrays[key], _ = self.load(data=val, **kwargs)

        return arrays

    def load_hdf5(self, fname, **kwargs):

        infos = kwargs['infos'] if 'infos' in kwargs else None

        return hdf5.load(fname, infos)

    @check_data_is_loaded
    def load_data_in_memory(self):
        
        pass

    @check_data_is_loaded
    def prepare_cohorts(self, fold, cohorts):
        """
        Method to separate out data into specific cohorts, the sampling rate of which
        will be defined in self.sampling_rates

        IMPORTANT: this is a default template; please modify as needed for your data

        """
        for k in cohorts:
            assert k in self.meta

        for split in ['train', 'valid']:

            # --- Determine mask corresponding to current split 
            if fold == -1:
                mask = np.ones(self.meta['valid'].size, dtype='bool')
            elif split == 'train': 
                mask = self.meta['valid'] != fold
            elif split == 'valid':
                mask = self.meta['valid'] == fold

            # --- Define cohorts based on cohorts lambda functions
            for key, f in cohorts.items():
                self.cohorts[split][key] = np.nonzero(f(self.meta) & mask)[0]

            # --- Randomize indices for next epoch
            for cohort in self.cohorts[split]:
                self.current[split][cohort] = {'epoch': -1, 'count': 0}
                self.prepare_next_epoch(split=split, cohort=cohort)

    @check_data_is_loaded
    def set_sampling_rates(self, rates={}):

        assert set(self.cohorts['train'].keys()) == set(rates.keys())
        assert sum(list(rates.values())) == 1

        keys = sorted(rates.keys())
        vals = [rates[k] for k in keys]

        lower = np.array([0] + vals[:-1])
        upper = np.array(vals[1:] + [1])

        self.sampling_rates = {
            'cohorts': keys,
            'lower': np.array(lower),
            'upper': np.array(upper)} 

    def set_training_rates(self, rates={'train': 0.8, 'valid': 0.2}):

        assert 'train' in rates
        assert 'valid' in rates

        self.training_rates = rates

    def prepare_next_epoch(self, split, cohort):

        assert cohort in self.cohorts[split]

        self.indices[split][cohort] = np.random.permutation(self.cohorts[split][cohort].size)
        self.current[split][cohort]['epoch'] += 1
        self.current[split][cohort]['count'] = 0 

    def prepare_next_array(self, split=None, cohort=None):

        if split is None:
            split = 'train' if np.random.rand() < self.training_rates['train'] else 'valid'

        if cohort is None:
            if self.sampling_rates is not None:
                i = np.random.rand()
                i = (i < self.sampling_rates['upper']) & (i >= self.sampling_rates['lower'])
                i = int(np.nonzero(i)[0])
                cohort = self.sampling_rates['cohorts'][i]
            else:
                cohort = sorted(self.cohorts[split].keys())[0]

        c = self.current[split][cohort]
        i = self.indices[split][cohort]

        if c['count'] > i.size - 1:
            self.prepare_next_epoch(split, cohort)
            c = self.current[split][cohort]
            i = self.indices[split][cohort]

        index = self.meta['index'][i[c['count']]]
        coord = self.meta['coord'][i[c['count']]]
        data = {k: '%s/%s' % (self.DS_PATH, v) for k, v in self.data[index].items()} 

        # --- Increment counter
        c['count'] += 1

        # --- Finalize meta
        meta = {'coord': coord, 'index': index, 'split': split, 'cohort': cohort}
        for k in ['mu', 'sd']:
            if k in self.meta:
                meta[k] = self.meta[k][index]

        return data, meta

    def get(self, shape, split=None, cohort=None):

        # --- Load data
        data, meta = self.prepare_next_array(split=split, cohort=cohort)
        arrays = self.load_dict(data=data, infos={
            'point': [meta['coord'], 0.5, 0.5], 
            'shape': shape})

        # --- Preprocess
        arrays = self.preprocess(arrays, meta)

        return arrays

    def preprocess(self, arrays, meta, **kwargs):

        return arrays

    def generator(self, shape, split, batch_size):
        """
        Method to wrap the self.get() method in a Python generator for training input

        """
        while True:

            xs, ys = [], []

            for i in range(batch_size):

                arrays = self.get(shape=shape, split=split) 
                xs.append(arrays['dat'])
                ys.append(arrays['lbl'][..., 0])

            yield np.stack(xs), np.stack(ys)

# ===================================================================================
# client = Client(DS_PATH='../../data')
# ===================================================================================
# client.make_summary(
#     query={
#         'dat': 'dat.hdf5',
#         'lbl': 'bet.hdf5'},
#     CLASSES=2)
# ===================================================================================
# client.prepare_cohorts(fold=0, cohorts={
#     1: lambda meta : meta[1] & ~meta[2],
#     2: lambda meta : meta[2]})
# client.set_sampling_rates(rates={
#     1: 0.5,
#     2: 0.5})
# ===================================================================================
# for i in range(500):
#     printp('Running iteration: %04i' % (i + 1), i / 499)
#     arrays = client.get(shape=[1, 512, 512])
# ===================================================================================
