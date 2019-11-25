import os, glob, pickle
import numpy as np
from dl_core.io import hdf5

class Client():

    def __init__(self, SUMMARY_PATH=None):

        self.data = None 
        self.meta = {} 

        self.cohorts = {'train': {}, 'valid': {}}
        self.indices = {'train': {}, 'valid': {}}
        self.current = {'train': {}, 'valid': {}}

        self.sampling_rates = None 
        self.set_training_rates()

        # --- Prepare summary path
        VAR = 'DL_CLIENT_SUMMARY_PATH'
        self.SUMMARY_PATH = SUMMARY_PATH if VAR not in os.environ else os.environ[VAR]
        if os.path.exists(self.SUMMARY_PATH or ''):
            self.load_summary()

    def check_data_is_loaded(func):
        """
        Method (decorator) to ensure the self.data / self.meta is loaded

        """
        def wrapper(self, *args, **kwargs):

            if self.data is None:
                self.print_ljust('WARNING data has not been loaded, will attempt load now', END='\n')
                self.load_summary()
                self.print_ljust('WARNING data has been successfully loaded', END='\r')

            return func(self, *args, **kwargs)

        return wrapper

    def make_summary(self, query, CLASSES=2, N_FOLDS=5, SUMMARY_PATH='./pkls/summary.pkl'):
        """
        Method to read all data and make summary dictionary 

        :params

          (dict) query : {

            'root': '/path/to/root/dir',
            'dat': [query_00],
            'lbl': [query_01], ...

          }

        :return

          (dict) summary : {

            'data': [{paths_00}, {paths_01}, ...],
            'meta': {
              'index': [0, 0, 0, ..., 1, 1, 1, ...],
              'coord': [0, 1, 2, ..., 0, 1, 2, ...],
              0: [1, 1, 0, 0, 1, ...],                  # presence of class == 0 at slice pos
              1: [1, 1, 0, 0, 1, ...],                  # presence of class == 1 at slice pos
              ...
            } 
          }

        IMPORTANT: this is a default template; please modify as needed for your data

        """
        assert 'root' in query
        assert len(query) > 1

        root = query.pop('root')
        keys = sorted(query.keys())

        q = query.pop(keys[0])
        matches = glob.glob('%s/**/%s' % (root, q), recursive=True)

        DATA = []
        META = {c: [] for c in range(CLASSES + 1)}
        META['index'] = []
        META['coord'] = []
        META['mu'] = []
        META['sd'] = []

        for n, m in enumerate(matches):

            self.print_ljust('CREATING SUMMARY (%07i/%07i): %s' % (n + 1, len(matches), m))

            d = {keys[0]: m}
            b = os.path.dirname(m)

            # --- Find other matches
            for key in keys[1:]:
                ms = glob.glob('%s/%s' % (b, query[key]))

                if len(ms) == 1: 
                    d[key] = ms[0]

                elif len(ms) == 0:
                    self.print_ljust('ERROR no match found: %s/%s' % (b, query[key]), END='\n')

                else: 
                    self.print_ljust('ERROR more than a single match found: %s/%s' % (b, query[key]), END='\n')

            # --- Caculate summary meta information
            if len(d) == len(keys):

                # --- Aggregate slice-by-slice label information
                if 'lbl' in d:
                    data, _ = self.load(d['lbl'])

                    for c in range(CLASSES + 1):
                        s = np.sum(data == c, axis=(1, 2, 3)) > 0
                        META[c].append(s)

                # --- Aggregate slice-by-slice data information
                if 'dat' in d:
                    data, _ = self.load(d['dat'])

                    META['mu'].append(data.mean(axis=(0, 1, 2)).reshape(1, -1))
                    META['sd'].append(data.std(axis=(0, 1, 2)).reshape(1, -1))

                # --- Aggregate index/coord information
                META['index'].append(np.ones(data.shape[0], dtype='int') * len(DATA))
                META['coord'].append(np.arange(data.shape[0]) / (data.shape[0] - 1))
                DATA.append(d)

        # --- Set validation fold (N-folds)
        valid = np.arange(len(META['index'])) % N_FOLDS
        valid = valid[np.random.permutation(valid.size)]
        META['valid'] = [np.ones(c.size) * v for c, v in zip(META['coord'], valid)]

        # --- Concatenate all vectors
        META = {k: np.concatenate(v) for k, v in META.items()}

        # --- Serialize
        PKL = self.SUMMARY_PATH or SUMMARY_PATH
        os.makedirs(os.path.dirname(PKL), exist_ok=True)
        pickle.dump({'data': DATA, 'meta': META}, open(PKL, 'wb'))

        self.data = DATA
        self.meta = META

        # --- Final output
        self.print_final('FOUND A TOTAL OF: %i PATIENTS | %i SLICES' % (len(DATA), len(META['coord'])))

    def load_summary(self, SUMMARY_PATH='./pkls/summary.pkl'):

        PKL = self.SUMMARY_PATH or SUMMARY_PATH
        if not os.path.exists(PKL):
            print('ERROR provided Pickle file not found: %s' % PKL)
            return

        d = pickle.load(open(PKL, 'rb'))
        self.data = d['data']
        self.meta = d['meta']

    def load(self, data, **kwargs):

        if type(data) is not str:
            return data

        LOAD_FUNC = {
            'hdf5': self.load_hdf5}

        ext = data.split('.')[-1]

        if ext in LOAD_FUNC:
            return LOAD_FUNC[ext](data, **kwargs)

        else:
            print('ERROR provided extension is not supported: %s' % ext)
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
        data = self.data[index]

        # --- Increment counter
        c['count'] += 1

        return data, {'coord': coord, 'index': index, 'split': split, 'cohort': cohort}

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

    def print_ljust(self, s, SIZE=80, END='\r'):

        print(s.ljust(SIZE), end=END)

    def print_final(self, s, SIZE=80):

        h = ''.join(['='] * SIZE)
        c = ''.join([' '] * 200)

        print(c, end='\r')
        print(h)
        self.print_ljust(s, SIZE=SIZE, END='\n')
        print(h)

# ===================================================================================
# client = Client(SUMMARY_PATH='../../data/pkls/summary.pkl')
# ===================================================================================
# client.make_summary(
#     query={
#         'root': '../../data/hdfs',
#         'dat': 'dat.hdf5',
#         'bet': 'bet.hdf5'},
#     LABELED='bet',
#     CLASSES=2)
# ===================================================================================
# client.load_summary()
# client.prepare_cohorts(fold=0, cohorts={
#     1: lambda meta : meta[1] & ~meta[2],
#     2: lambda meta : meta[2]})
# client.set_sampling_rates(rates={
#     1: 0.5,
#     2: 0.5})
# ===================================================================================
# for i in range(10000):
#     print('Running iteration: %04i' % i, end='\r')
#     x, y = client.get()
# ===================================================================================
