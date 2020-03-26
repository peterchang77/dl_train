import os
from dl_utils.general import *
from dl_utils.datasets import download
from ..client import Client 

def prepare(name, configs=None, keyword='', version_id=None):
    """
    Method to create Python generators for train / valid data

    """
    os.environ['JARVIS_PROJECT_ID'] = name
    if version_id is not None:
        os.environ['JARVIS_VERSION_ID'] = version_id
    pattern = 'client*' + keyword

    client = Client(configs=configs, pattern=pattern)

    gen_train, gen_valid = client.create_generators()

    return gen_train, gen_valid, client
