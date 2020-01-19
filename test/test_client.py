import unittest
import os, shutil
from dl_train.client import Client
from dl_utils.general import overload

@overload(Client)
def preprocess(self, arrays, **kwargs):

    # arrays['xs']['dat'] = arrays['xs']['dat'].clip(min=0, max=256) / 64

    return arrays

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

        # --- Test 100 random loads 
        self.client.test(100)

    @classmethod
    def tearDownClass(self):
        
        pass

if __name__ == '__main__':

    unittest.main()

    pass
