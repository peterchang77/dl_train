import os, glob, json
import numpy as np, pandas as pd
from dl_utils import io
from dl_utils.db import DB, funcs 
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

        self.indices = {'train': {}, 'valid': {}}
        self.current = {'train': {}, 'valid': {}}

        self.specs = None
        self.infos = {} 
        self.tiles = tuple([False] * 4) 
        self.apply = None

        self.sampling_rates = None 
        self.set_training_rates()

        from_json = kwargs.get('from_json', './client.json')
        if os.path.exists(from_json):
            configs = json.load(open(from_json, 'r'))
            for key, config in configs.items():
                setattr(self, key, config)

        if os.path.exists(self.DS_FILE):
            printd('Loading client')
            self.load_csv()

    def to_json(self, fname='./client.json'):
        """
        Method to serialize key attr as JSON

        """
        configs = {}

        for attr in ['current', 'specs', 'infos', 'tiles']:
            configs[attr] = getattr(self, attr)

        json.dump(configs, open(fname, 'w'))

    def parse_args(self, *args, **kwargs):
        """
        Method to parse arguments into final kwargs dict 

        """
        DEFAULTS = {
            'DS_PATH': '',
            'DS_FILE': None}

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

        kwargs = {**DEFAULTS, **env, **args_, **kwargs}

        kwargs['DS_FILE'] = kwargs['DS_FILE'] or \
            '{}/csvs/summary.csv.gz'.format(kwargs['DS_PATH'])

        return kwargs

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
            self.df = pd.read_csv(DS_FILE, index_col='sid')
            self.determine_loadable_columns()

    def to_csv(self, DS_FILE=None):

        DS_FILE = DS_FILE or self.DS_FILE
        os.makedirs(os.path.dirname(DS_FILE), exist_ok=True)
        self.df.to_csv(DS_FILE)

    def determine_loadable_columns(self, cohorts=None):
        """
        Method to determine loadable columns (e.g. valid file names)

        """
        self.loadable_columns = [c for c, f in self.df.iloc[0].items() if os.path.exists(str(f))]

    def make_summary(self, query=None, db=None, funcs_def='mr_train', join=None, folds=5, DS_FILE=None, **kwargs):
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
        kwargs = funcs.init(funcs_def, load=io.load, **kwargs)
        df = db.apply(**kwargs)

        # --- Create series (fnames + valid)
        series = db.df_merge(rename=False)
        valids = np.arange(series.shape[0]) % folds 
        series['valid'] = valids[np.random.permutation(valids.size)]

        # --- Join series (fnames + valid)
        join = (join or db.fnames.columns.tolist()) + ['valid']
        cols = join + df.columns.tolist()
        df = df.join(series[join])
        df = df[cols]

        # --- Serialize
        self.df = df
        self.to_csv(DS_FILE)

        # --- Final output
        printd('Summary complete: %i patients | %i slices' % (series.shape[0], df.shape[0]))

    def load(self, row, **kwargs):

        arrays = {'xs': {}, 'ys': {}}
        for k in arrays:
            for key, spec in self.specs[k].items():

                # --- Load from file 
                if spec['loads'] in self.loadable_columns:
                    infos = self.get_infos(row, spec['shape']) 
                    arrays[k][key] = io.load(row[spec['loads']], infos=infos)[0]

                # --- Load from row
                else:
                    if spec['loads'] is not None:
                        arrays[k][key] = np.array(row[spec['loads']])
                    else:
                        arrays[k][key] = np.ones(spec['shape'], dtype=spec['dtype'])

        return arrays

    @check_data_is_loaded
    def load_data_in_memory(self):
        
        pass

    @check_data_is_loaded
    def prepare_cohorts(self, fold, cohorts):
        """
        Method to separate out data into specific cohorts for stratified sampling.

        :params

          (int) fold    : fold to use as validation 
          (vec) cohorts : boolean vector equal in size to self.df.shape[0]

        Note the sampling rate is defined in self.sampling_rates.

        IMPORTANT: this is a default template; please modify as needed for your data

        """
        printd('Preparing client')

        for split in ['train', 'valid']:

            # --- Determine mask corresponding to current split 
            if fold == -1:
                mask = np.ones(self.df.shape[0], dtype='bool')
            elif split == 'train': 
                mask = self.df['valid'] != fold
                mask = mask.to_numpy()
            elif split == 'valid':
                mask = self.df['valid'] == fold
                mask = mask.to_numpy()

            # --- Define cohorts based on cohorts lambda functions
            for key, s in cohorts.items():
                self.indices[split][key] = np.nonzero(np.array(s) & mask)[0]

            for cohort in self.indices[split]:

                # --- Randomize indices for next epoch
                if cohort not in self.current[split]:
                    self.current[split][cohort] = {'epoch': -1, 'count': 0}
                    self.prepare_next_epoch(split=split, cohort=cohort)

                # --- Reinitialize old index
                else:
                    self.shuffle_indices(split, cohort)
    
    @check_data_is_loaded
    def print_cohorts(self):
        """
        Method to generate summary of cohorts

        ===========================
        TRAIN
        ===========================
        cohort-A: 1000
        cohort-B: 1000
        ...
        ===========================
        VALID
        ===========================
        cohort-A: 1000
        cohort-B: 1000
        ...

        """
        keys = sorted(self.indices['train'].keys())

        for split in ['train', 'valid']:
            printb(split.upper())
            for cohort in keys:
                size = self.indices[split][cohort].size
                printd('{}: {:06d}'.format(cohort, size))

    @check_data_is_loaded
    def set_sampling_rates(self, rates={}):

        printd('Setting sampling rates')

        assert set(self.indices['train'].keys()) == set(rates.keys())
        assert sum(list(rates.values())) == 1

        keys = sorted(rates.keys())
        vals = [rates[k] for k in keys]
        vals = [sum(vals[:n]) for n in range(len(vals) + 1)]

        lower = np.array(vals[:-1])
        upper = np.array(vals[1:])

        self.sampling_rates = {
            'cohorts': keys,
            'lower': np.array(lower),
            'upper': np.array(upper)} 

    def set_training_rates(self, rates={'train': 0.8, 'valid': 0.2}):

        assert 'train' in rates
        assert 'valid' in rates

        self.training_rates = rates

    def set_specs(self, specs, tiles=(False, False, False), yml_file=None):

        assert 'xs' in specs
        assert 'ys' in specs

        self.specs = specs
        self.tiles = tuple(list(tiles) + [False])

    def get_specs(self):

        extract = lambda x : {
            'shape': [None if t else s for s, t in zip(x['shape'], self.tiles)],
            'dtype': x['dtype']}

        specs_ = {'xs': {}, 'ys': {}} 
        for k in specs_:
            for key, spec in self.specs[k].items():
                specs_[k][key] = extract(spec)

        return specs_

    def get_infos(self, row, shape):

        infos_ = self.infos.copy()
        infos_['point'] = [row['coord'], 0.5, 0.5]
        infos_['shape'] = shape[:3]

        return infos_

    def prepare_next_epoch(self, split, cohort):

        assert cohort in self.indices[split]

        # --- Increment current
        self.current[split][cohort]['epoch'] += 1
        self.current[split][cohort]['count'] = 0 
        self.current[split][cohort]['rseed'] = np.random.randint(2 ** 32)

        # --- Random shuffle
        self.shuffle_indices(split, cohort)

    def shuffle_indices(self, split, cohort):

        # --- Seed
        np.random.seed(self.current[split][cohort]['rseed'])

        # --- Shuffle indices
        s = self.indices[split][cohort].size
        p = np.random.permutation(s)
        self.indices[split][cohort] = self.indices[split][cohort][p]

    def prepare_next_array(self, split=None, cohort=None, row=None):

        if row is not None:
            return {'row': row, 'split': split, 'cohort': cohort}

        if split is None:
            split = 'train' if np.random.rand() < self.training_rates['train'] else 'valid'

        if cohort is None:
            if self.sampling_rates is not None:
                i = np.random.rand()
                i = (i < self.sampling_rates['upper']) & (i >= self.sampling_rates['lower'])
                i = int(np.nonzero(i)[0])
                cohort = self.sampling_rates['cohorts'][i]
            else:
                cohort = sorted(self.indices[split].keys())[0]

        c = self.current[split][cohort]

        if c['count'] > self.indices[split][cohort].size - 1:
            self.prepare_next_epoch(split, cohort)
            c = self.current[split][cohort]

        ind = self.indices[split][cohort][c['count']]
        row = self.df.iloc[ind]

        # --- Increment counter
        c['count'] += 1

        return {'row': row, 'split': split, 'cohort': cohort} 

    def get(self, split=None, cohort=None, row=None):

        # --- Load data
        kwargs = self.prepare_next_array(split=split, cohort=cohort, row=row)
        arrays = self.load(**kwargs)

        # --- Preprocess
        arrays = self.preprocess(arrays, **kwargs)

        # --- Ensure that spec matches
        for k in ['xs', 'ys']:
            for key in arrays[k]:
                shape = self.specs[k][key]['shape']
                dtype = self.specs[k][key]['dtype']
                arrays[k][key] = arrays[k][key].reshape(shape).astype(dtype)

        return arrays

    def test_all(self, lo=0, hi=None):
        """
        Method to test self.get() method for all rows

        """

        columns = ['sid'] + list(self.df.columns)
        to_dict = lambda row : {k: row[n] for n, k in enumerate(columns)}

        count = 0
        hi = None or self.df.shape[0]

        for row in self.df.iloc[lo:hi].itertuples():

            self.get(row=to_dict(row))
            count += 1
            printp('Loading all rows | {:06d}'.format(count), count / self.df.shape[0])

    def test_get(self, n=1000):
        """
        Method to test self.get() method

        """
        for i in range(n):
            printp('Loading | {:04d}'.format(i), (i + 1) / n)
            self.get()

        printd('Completed {} self.get() iterations successfully'.format(n))

    def preprocess(self, arrays, **kwargs):

        return arrays 

    def generator(self, split, batch_size):
        """
        Method to wrap the self.get() method in a Python generator for training input

        """
        while True:

            xs = []
            ys = []

            for i in range(batch_size):

                arrays = self.get(split=split) 
                xs.append(arrays['xs'])
                ys.append(arrays['ys'])

            xs = {k: np.stack([x[k] for x in xs]) for k in self.specs['xs']}
            ys = {k: np.stack([y[k] for y in ys]) for k in self.specs['ys']}

            yield xs, ys

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
