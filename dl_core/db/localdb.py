import os, yaml, pandas as pd 

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

    def initialize(self, csv_file, refresh_rows=False, refresh_cols=True, **kwargs):
        """
        Method to initialize CSV-backed Cache DB

          (1) Populate rows (fnames)
          (2) Populate cols (meta functions)

        """
        df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
        self.configs = configs

    def cursor(self):
        """
        Method to create Python generator to iterate through dataset
        
        """
        for sid, row in self.df.iterrows():

            fnames = {k: v for k, v in row.items() if k.find('fname') > -1}

            yield sid, fnames
