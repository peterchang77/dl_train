import os, yaml, pandas as pd 
from dl_core.utils import * 

# ===================================================================
# OVERVIEW
# ===================================================================
# 
# Object that generates a cached view of the underlying raw data that
# is stored in the file system as serialized Jarvis objects.
# 
# The intermediate cached view is instantiated either as a CSV file
# (e.g. pandas DataFrame) or as a MongoDB store.
# 
# The cached view must be able to support the following actions:
# 
#   * query()
#   * iterate()
#   * to_json()
#   * to_csv()
# 
# ===================================================================
# 
#                     [ CSV FILE ] [ MONGO DB ]
#                              \  / 
#                               \/
#       [ LOCAL FILE-SYSTEM ] <----> [ EXTERNAL CLIENT ] 
# 
# ===================================================================

class CacheDB():

    def __init__(self, configs=None, yml_file=None):
        """
        Method to initialize Cache DB

        """
        configs = configs or {}
        if yml_file is not None:
            configs.update(yaml.load(open(yml_file), Loader=yaml.FullLoader))

        # --- Save attributes
        self.configs = configs
        self.columns = configs.get('columns', []) 

        self.initialize(**configs)

    def update_all(self):
        """
        Method to re-index all files

        """
        # --- Query for all matching data

        # --- Iterate through each match with self.update_row()

    def update_row(self, sid, arrs, **kwargs):
        """
        Method to update db row with current arrays

        NOTE: add custom columns by registered methods in UPDATE_FUNCTIONS dict

        """
        for key, func in self.UPDATE_FUNCTIONS.items():
            func(sid, arrs, **kwargs)

    def query(self):
        """
        Method to query db for data

        """
        pass

    def iterate(self, func):
        """
        Method to iterate through data and apply provided func

        """
        pass

class CacheCSV(CacheDB):

    def initialize(self, csv_file, refresh_sids=False, refresh_cols=True, **kwargs):
        """
        Method to initialize CSV-backed Cache DB

          (1) Populate rows (fnames)
          (2) Populate cols (meta functions)

        """
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, index_col='sid')
        else: 
            df = pd.DataFrame()
            df.index.name = 'sid'

        # --- Split df into sids and meta
        self.df = self.df_split(df)
        
        # --- Refresh sids 
        if self.df['sids'].shape[0] == 0 or refresh_sids:
            self.refresh_sids()

        # --- Refresh cols
        if self.df['sids'].shape[0] != self.df['meta'].shape[0] or refresh_cols:
            pass

    def df_split(self, df):
        """
        Method to split DataFrame into `sids` + `meta`

        """
        split = {}
        split['sids'] = df[[k for k in df if k[:6] == 'fname-']]
        split['meta'] = df[[k for k in df if k not in split['sids']]]

        # --- Rename `fnames-`
        split['sids'] = split['sids'].rename(columns={k: k[6:] for k in split['sids'].columns})

        return split

    def df_merge(self):
        """
        Method to merge DataFrame

        """
        # --- Rename `fnames-`
        c = {k: 'fname-%s' % k for k in self.df['sids'].columns}

        return pd.concat((self.df['sids'].rename(columns=c), self.df['meta']), axis=1, sort=True)

    def refresh_sids(self):
        """
        Method to refresh sids by updating with results of query

        """
        matches, _ = find_matching_files(self.configs['query'], verbose=False)
        self.df['sids'] = pd.DataFrame.from_dict(matches, orient='index')

        # --- Propogate indices if meta is empty 
        if self.df['meta'].shape[0] == 0:
            self.df['meta'] = pd.DataFrame(index=self.df['sids'].index)

        self.df['sids'].index.name = 'sid'
        self.df['meta'].index.name = 'sid'

    def refresh_cols(self):
        """
        Method to refresh cols

        """
        # --- Update columns
        for k in self.columns:
            if k not in self.df['meta']:
                self.df['meta'] = None

        # --- Find rows with a None column entry

        # --- Update rows

    def cursor(self):
        """
        Method to create Python generator to iterate through dataset
        
        """
        for sid, fnames in self.df['sids'].iterrows():
            yield sid, fnames

    def to_json(self, max_rows=None):
        """
        Method to serialize contents of DB to JSON

        :return 
        
          (dict) combined = {

            [sid_00]: {
                'fnames': {'dat': ..., 'lbl': ....},
                'meta_00': ...,
                'meta_01': ..., }, 

            [sid_01]: {
                'fnames': {'dat': ..., 'lbl': ....},
                'meta_00': ...,
                'meta_01': ...,}, 
            ...

        }

        """
        meta = self.df['meta'].to_dict(orient='index')
        sids = self.df['sids'].to_dict(orient='index')
        sids = {k: {'fnames': v} for k, v in sids.items()}

        return {k: {**sids[k], **meta[k]} for k in sids}

    def to_csv(self, csv=None):
        """
        Method to serialize contents of DB to CSV

        """
        csv = csv or self.configs.get('csv_file', None)
        if csv is not None:
            df = self.df_merge()
            df.to_csv(csv)

# ===============================================
# c = CacheCSV(yml_file='./configs.yml')
# ===============================================
