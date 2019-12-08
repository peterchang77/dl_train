# Overview

This repository contains a shared toolbox of code and tutorials for the application of deep learning neural networks to medical imaging. This includes the main `dl_core` module which contains several utility libraries (i/o, visualization, etc) as well as an independent series of tutorials complete with downloadable datasets written as Jupyter notebooks. For those interested in the tutorials, it is recommended to run these directly in Google Colaboratory; more detailed information is available below.

1. [Preparing Environment](#preparing-environment) 
2. [Tutorials](#tutorials) 

# Preparing Environment

As above, it is recommended to start by launching tutorials directly in Google Colaboratory. Within these tutorials, the necessary lines of code to dynamically set your development environment are provided and should work without modification. For instructions, go directly to [Tutorials](#tutorials) below. 

If you decide to run this code and/or the `dl_core` library on local compute (or a personal cloud instance) you must prepare your development environment. Generically, this will include:

1. Cloning a copy of this repository and adding the repository root directory to `$PYTHONPATH`.
2. *(optional)* Downloading any required data for a specific tutorial.
3. *(optional)* Set the requisite shell environment variables to make dataset visible to Python `client`.

## Using the `setenv.py` script

To help configure your environment and/or download data, a `setenv.py` script is available within the root of this directory. To use this script: 

1. Obtain a copy of the `setenv.py` file. This can be done either via cloning this repository locally, or simply via `wget` on Linux machines:

`wget -O setenv.py https://raw.githubusercontent.com/peterchang77/dl_core/master/setenv.py`  

2. Invoke the script **at the beginning** of your code (before other imports):

```
from setenv import prepare_environment

prepare_environment(
    DL_PATH=[path to local repository e.g. '/home/peter/python/dl_core'],
    DS_PATH=[path to local dataset e.g. '/data/raw/brats'],
    PK_FILE=[path to local summary Pickle file for given dataset e.g. '/data/raw/brats/pkls/summary.pkl']
    DS_NAME=[name of dataset to download e.g. 'brats'],
    ignore_existing=[False/True],
    CUDA_VISIBLE_DEVICES=[None/0/1/2...])
```

The following arguments may be passed:

* `DL_PATH`: complete path to the local `dl_core` library (location where this GitHub repository is located); if the repository has not yet been downloaded, this path represents the destination where the GitHub repository will be cloned; by default the `dl_core` library will be downloaded to: `$HOME/python/dl_core`

* `DS_PATH`: complete path to the dataset used in a given tutorial (or your current experiment); if the dataset has not yet been downloaded, this path represents the destination where data will be archived and unzipped locally; by default the dataset will be downloaded to: `/data/raw/$DS_NAME`

* `PK_FILE`: complete path to the local summary Pickle file for a given dataset (please see data client tutorial for more information); by default the Pickle file will be located at: `/data/raw/$DS_NAME/pkls/summary.pkl`

* `DS_NAME`: name of dataset to be downloaded

* `ignore_existing`: force the dataset to be re-downloaded even if available locally

* `CUDA_VISIBLE_DEVICES`: the GPU device to use for training if multiple are available on machine

## Manual configuration

For persistent configuration between Python / Jupyter sessions, it is recommended to simply set the required paths as shell environment variables: `$DS_PATH`, `$PK_FILE`. In addition, the location of the `dl_core` repository (`$DL_PATH`) can be added directly to the `$PYTHONPATH` variable. Given this, the following demonstrates example lines that can be added to `~/.bashrc` (or any other preferred method to set shell environment variables):

```
# Set dl_core library paths
export DL_PATH=/home/peter/python/dl_core
export PYTHONPATH=$PYTHONPATH:$DL_PATH
export DS_PATH=/data/raw/brats
export DS_FILE=/data/raw/brats/pkls/summary.pkl
```

Note that `CUDA_VISIBLE_DEVICES` is recommended to be set dynamically to allocate GPU resources as needed.

# Tutorials

Any `*.ipynb` tutorial can be run directly from GitHub in a Google Colab hosted Jupyter instance using the following URL pattern:

`https://https://colab.research.google.com/github/peterchang77/dl_core/blob/master/docs/notebooks/[name-of-notebook.ipynb]`

Here is an overview of the currently available notebooks with direct Google Colab links:

* Introduction to Tensorflow / Keras 2.0 API [link](https://bit.ly/37U4noo) 
* Using HDF5 File Format [link](https://bit.ly/33yY1rd)
* How to Create a Data Client [link](https://bit.ly/2Y28ydn) 
* Overview of U-Net [link](https://bit.ly/35PHxwk)
