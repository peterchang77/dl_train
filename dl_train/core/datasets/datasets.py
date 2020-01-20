from dl_utils.general import *
from dl_utils.datasets import download
from ..client import Client 

def prepare(name, configs, path='/data/raw', custom_client=None):
    """
    Method to create Python generators for train / valid data

    """
    client = '{}/{}/ymls/client.yml'.format(path, name)
    client = Client(client=client, configs=configs)

    return client.create_generators()
