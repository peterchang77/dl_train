import os, glob, yaml
import numpy as np, pandas as pd
from dl_utils import io
from dl_utils.db import DB
from dl_utils.general import *

class Client():

    def __init__(self, client='./client.yml', configs=None, *args, **kwargs):
        """
        Method to initialize client

        """
        # --- Serialization attributes
        self.ATTRS = ['_db', 'current', 'batch', 'specs']

        # --- Initialize existing settings from *.yml
        self.load_yml(client, configs)

        # --- Initialize db
        self.db = DB(*(*args, *(self._db,)), **kwargs)
        self._db = self.db.get_files()['yml'] or self.db.get_files()['csv']

        # --- Initialize batch composition 
        self.prepare_batch()

        # --- Initialize normalization functions
        self.init_normalization()

    def load_yml(self, client, configs):
        """
        Method to load metadata from YML

        """
        DEFAULTS = {
            '_db': None,
            'indices': {'train': {}, 'valid': {}},
            'current': {'train': {}, 'valid': {}},
            'batch': {'fold': -1, 'size': None, 'sampling': None, 'training': {'train': 0.8, 'valid': 0.2}},
            'specs': {'xs': {}, 'ys': {}, 'infos': {}, 'tiles': [False] * 4}}

        configs = configs or {}
        if os.path.exists(client):
            with open(client, 'r') as y:
                configs = {**yaml.load(y, Loader=yaml.FullLoader), **configs}

        # --- Initialize default values
        for key, d in DEFAULTS.items():
            configs[key] = {**DEFAULTS[key], **configs.get(key, {})} if type(d) is dict else \
                configs[key] or DEFAULTS[key]

        # --- Set attributes
        for attr, config in configs.items():
            setattr(self, attr, config)

    def to_dict(self):
        """
        Method to create dictionary of metadata

        """
        return {attr: getattr(self, attr) for attr in self.ATTRS}

    def to_yml(self, fname='./client.yml'):
        """
        Method to serialize metadata to YML 

        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w') as y:
            yaml.dump(self.to_dict(), y)

    def check_data_is_loaded(func):
        """
        Method (decorator) to ensure the self.data / self.meta is loaded

        """
        def wrapper(self, *args, **kwargs):

            if self.db.fnames.size == 0:
                self.db.load_csv()

            return func(self, *args, **kwargs)

        return wrapper

    @check_data_is_loaded
    def load_data_in_memory(self):
        
        pass

    @check_data_is_loaded
    def prepare_batch(self, fold=None, sampling_rates=None, training_rates={'train': 0.8, 'valid': 0.2}):
        """
        Method to prepare composition of data batches

        :params

          (int)  fold           : fold to use as validation 
          (dict) sampling_rates : rate to load each stratified cohort
          (dict) training_rates : rate to load from train / valid splits

        """
        # --- Set rates
        self.set_training_rates(training_rates)
        self.set_sampling_rates(sampling_rates)

        # --- Set default fold
        fold = fold or self.batch['fold']

        for split in ['train', 'valid']:

            # --- Determine mask corresponding to current split 
            if fold == -1:
                mask = np.ones(self.db.header.shape[0], dtype='bool')
            elif split == 'train': 
                mask = self.db.header['valid'] != fold
                mask = mask.to_numpy()
            elif split == 'valid':
                mask = self.db.header['valid'] == fold
                mask = mask.to_numpy()

            # --- Find indices for current cohort / split 
            for key in self.batch['sampling']:
                self.indices[split][key] = np.nonzero(np.array(self.db.header[key]) & mask)[0]

            for cohort in self.indices[split]:

                # --- Randomize indices for next epoch
                if cohort not in self.current[split]:
                    self.current[split][cohort] = {'epoch': -1, 'count': 0}
                    self.prepare_next_epoch(split=split, cohort=cohort)

                # --- Reinitialize old index
                else:
                    self.shuffle_indices(split, cohort)
    
    def set_training_rates(self, rates={'train': 0.8, 'valid': 0.2}):

        assert 'train' in rates
        assert 'valid' in rates

        self.batch['training'] = rates

    def set_sampling_rates(self, rates=None):

        rates = rates or self.batch['sampling']

        # --- Default, all cases without stratification
        if rates is None:
            self.db.header['all'] = True
            rates = {'all': 1.0}

        if 'all' in rates and 'all' not in self.db.header:
            self.db.header['all'] = True

        assert sum(list(rates.values())) == 1

        keys = sorted(rates.keys())
        vals = [rates[k] for k in keys]
        vals = [sum(vals[:n]) for n in range(len(vals) + 1)]

        lower = np.array(vals[:-1])
        upper = np.array(vals[1:])

        self.batch['sampling'] = rates

        self.sampling_rates = {
            'cohorts': keys,
            'lower': np.array(lower),
            'upper': np.array(upper)} 

    def init_normalization(self):
        """
        Method to initialize normalization functions as defined by self.spec

        arr = (arr.clip(min, max) - shift) / scale

        There three methods for defining parameters in *.yml file:

          (1) shift: 64      ==> use raw value if provided
          (2) shift: 'mu'    ==> use corresponding column in DataFrame
          (3) shift: '$mean' ==> use corresponding Numpy function

        """
        self.norm_lambda = {'xs': {}, 'ys': {}}
        self.norm_kwargs = {'xs': {}, 'ys': {}}

        # --- Lambda function for extracting kwargs
        extract = lambda x, row, arr : row[x] if x[0] != '@' else getattr(np, x[1:])(arr)

        for a in ['xs', 'ys']:
            for key, specs in self.specs[a].items():
                if specs['norms'] is not None:

                    norms = specs['norms']

                    # --- Set up appropriate lambda function
                    if 'clip' in norms and ('shift' in norms or 'scale' in norms):
                        l = lambda x, clip, shift=0, scale=1 : (x.clip(**clip) - shift) / scale

                    elif 'clip' in norms:
                        l = lambda x, clip : x.clip(**clip)

                    else:
                        l = lambda x, shift=0, scale=1 : (x - shift) / scale

                    self.norm_lambda[a][key] = l

                    # --- Set up appropriate kwargs function 
                    self.norm_kwargs[a][key] = lambda row, arr, norms : \
                        {k: extract(v, row, arr) if type(v) is str else v for k, v in norms.items()}

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

    def set_specs(self, specs):

        self.specs.update(specs)

    def get_specs(self):

        extract = lambda x : {
            'shape': [None if t else s for s, t in zip(x['shape'], self.specs['tiles'])],
            'dtype': x['dtype']}

        specs_ = {'xs': {}, 'ys': {}} 
        for k in specs_:
            for key, spec in self.specs[k].items():
                specs_[k][key] = extract(spec)

        return specs_

    def get_inputs(self, Input):
        """
        Method to create dictionary of Keras-type Inputs(...) based on self.specs

        """
        specs = self.get_specs()

        return {k: Input(
            shape=specs['xs'][k]['shape'],
            dtype=specs['xs'][k]['dtype'],
            name=k) for k in specs['xs']}

    def load(self, row, **kwargs):

        arrays = {'xs': {}, 'ys': {}}
        for k in arrays:
            for key, spec in self.specs[k].items():

                # --- Load from file 
                if spec['loads'] in self.db.fnames.columns:
                    infos = self.get_infos(row, spec['shape']) 
                    arrays[k][key] = io.load(row[spec['loads']], infos=infos)[0]

                # --- Load from row
                else:
                    arrays[k][key] = np.array(row[spec['loads']]) if spec['loads'] is not None else \
                        np.ones(spec['shape'], dtype=spec['dtype'])

        return arrays

    def get_infos(self, row, shape):

        infos_ = self.specs['infos'].copy()
        infos_['point'] = [row.get('coord', 0.5), 0.5, 0.5]
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
            row = self.db.row(row)
            return {'row': row, 'split': split, 'cohort': cohort}

        if split is None:
            split = 'train' if np.random.rand() < self.batch['training']['train'] else 'valid'

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
        row = self.db.row(ind)

        # --- Increment counter
        c['count'] += 1

        return {'row': row, 'split': split, 'cohort': cohort} 

    def get(self, split=None, cohort=None, row=None):

        # --- Load data
        kwargs = self.prepare_next_array(split=split, cohort=cohort, row=row)
        arrays = self.load(**kwargs)

        # --- Preprocess
        arrays = self.preprocess(arrays, **kwargs)

        # --- Normalize
        arrays = self.normalize(arrays, **kwargs)

        # --- Ensure that spec matches
        for k in ['xs', 'ys']:
            for key in arrays[k]:
                shape = self.specs[k][key]['shape']
                dtype = self.specs[k][key]['dtype']
                arrays[k][key] = arrays[k][key].reshape(shape).astype(dtype)

        return arrays

    def preprocess(self, arrays, **kwargs): 
        """
        Method to add custom preprocessing algorithms to data

        """
        return arrays

    def normalize(self, arrays, row, **kwargs):
        """
        Method to normalize data based on lambda defined set in self.norm_lambda

        """
        for a in ['xs', 'ys']:
            for key, func in self.norm_lambda[a].items():
                kwargs = self.norm_kwargs[a][key](row, arrays[a][key], self.specs[a][key]['norms'])
                arrays[a][key] = func(arrays[a][key], **kwargs)

        return arrays

    def test(self, n=None, split=None, cohort=None, aggregate=False):
        """
        Method to test self.get() method

        :params

          (int) n     : number of iterations; if None, then all rows
          (int) lower : lower bounds of row to load
          (int) upper : upper bounds of row to load

        """
        if aggregate:
            keys = lambda x : {k: [] for k in self.specs[x]}
            arrs = {'xs': keys('xs'), 'ys': keys('ys')} 

        # --- Iterate
        for i in range(n):

            printp('Loading iteration: {:06d}'.format(i), (i + 1) / n)
            arrays = self.get(split=split, cohort=cohort)

            if aggregate:
                for k in arrays:
                    for key in arrs[k]:
                        arrs[k][key].append(arrays[k][key][int(arrays[k][key].shape[0] / 2)])

        printd('Completed {} self.get() iterations successfully'.format(n), ljust=140)

        if aggregate:
            stack = lambda x : {k: np.stack(v) for k, v in x.items()}
            return {'xs': stack(arrs['xs']), 'ys': stack(arrs['ys'])}

    def generator(self, split, batch_size=None):
        """
        Method to wrap the self.get() method in a Python generator for training input

        """
        batch_size = batch_size or self.batch['size']
        if batch_size is None:
            printd('ERROR batch size must be provided if not already set')

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

    def create_generators(self, batch_size=None):

        gen_train = self.generator('train', batch_size)
        gen_valid = self.generator('valid', batch_size)

        return gen_train, gen_valid

# ===================================================================================
# client = Client()
# ===================================================================================
# client.make_summary(
#     query={
#         'dat': 'dat.hdf5',
#         'lbl': 'bet.hdf5'},
#     CLASSES=2)
# ===================================================================================
