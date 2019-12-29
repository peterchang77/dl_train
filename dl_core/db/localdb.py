import os, yaml, pandas as pd 
from dl_core.utils import find_matching_files

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

    def to_json(self, max_rows=None):
        """
        Method to serialize contents of DB to JSON

        """
        pass

    def to_csv(self):
        """
        Method to serialize contents of DB to CSV

        """
        pass

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
        df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()

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
        split['sids'] = df[[k for k in df if k.find('fname') > -1]]
        split['meta'] = df[[k for k in df if k not in split['sids']]]

        return split

    def df_merge(self, df):
        """
        Method to merge DataFrame

        """
        pass

    def refresh_sids(self):
        """
        Method to refresh sids by updating with results of query

        """
        matches = find_matching_files(self.configs['query'])
        self.df['sids'] = pd.DataFrame.from_dict(matches, orient='index')

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
