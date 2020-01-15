import unittest
import os, shutil
from dl_train.client import Client

class TestClient(unittest.TestCase):
    """
    Class to test DB() object functionality

    """
    @classmethod
    def setUpClass(self):
        """
        Set up test DB() object

        """
        # --- Create client
        self.client = Client('./client.yml')

    def test_get(self):

        # --- Create summary

    @classmethod
    def tearDownClass(self):
        
        pass

if __name__ == '__main__':

    unittest.main()

