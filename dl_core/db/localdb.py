import yaml, pandas as pd

class LocalDB():
    """
    LocalDB may be backed by:

      * None (in-memory structure generated from files upon initialization)
      * CSV file
      * MongoDB

    """

    def __init__(self, yml_path):
        """
        Method to initialize a new LocalDB object 

        """
        self.UPDATE_FUNCTIONS = {}

    def create_index(self):
        """
        Method to create data index

        """
        # --- Query for all matching data

        # --- Iterate through each match with self.update_row()
        pass

    def update_row(self, sid, arrs, **kwargs):
        """
        Method to update db row with current arrays

        NOTE: add custom columns by registered methods in UPDATE_FUNCTIONS dict

        """
        for key, func in self.UPDATE_FUNCTIONS.items():
            func(sid, arrs, **kwargs)

    def to_json(self):
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
