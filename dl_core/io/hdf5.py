import os, shutil, h5py
import numpy as np

def save(fname, data, meta={}, chunks=None, compression=None):
    """
    Method to save data and affine matrix in HDF5 format

    :params

      (str)       fname : path to file
      (np.ndarray) data : a 4D Numpy array
      (dict)       meta : {'affine': ...}
      (tuple)    chunks : (z, y, x, c) shape of chunks; by default 1 x Y x X x C
      (str) compression : either 'gzip' or 'lzf'; by default no compression

    """
    # --- Initialize
    fname, data, meta, chunks, compression = \
        init_save(fname, data, meta, chunks, compression)

    # --- Temporary file if fname already exists
    exists = os.path.exists(fname)
    if exists:
        fname = fname[:-5] + '_.hdf5'

    # --- Make required folder(s) 
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # --- Save file
    with h5py.File(fname, 'w') as f:

        kwargs = {'name': 'data', 'data': data, 'chunks': chunks}
        if compression is not None:
            kwargs['compression'] = compression

        f.create_dataset(**kwargs)

        # --- Save attributes
        for k, v in meta.items():
            f.attrs[k] = v 

    # --- Overwrite existing file 
    if exists:
        shutil.move(src=fname, dst=fname[:-6] + '.hdf5')

def init_save(fname, data, meta, chunks, compression):
    """
    Method to initialize save parameters to default values if needed

    Note that to properly initialize affine matrix, it is recommended to load data first via an Array() object

    """
    # --- Data checks
    assert type(fname) is str, 'Error fname is not str'
    if fname[-5:] != '.hdf5':
        fname += '.hdf5'

    assert type(data) is np.ndarray, 'Error data is not a NumPy array'
    assert data.ndim == 4, 'Error data is not a 4D array'

    if str(data.dtype) not in ['int16', 'uint8']:
        print('Warning data dtype is %s' % data.dtype)

    # --- Initialize new meta with desired keys
    keys = ['affine']
    meta = {k: meta[k] for k in keys if k in meta}

    if 'affine' not in meta:
        meta['affine'] = np.eye(4, dtype='float32')

    if type(meta['affine']) is list:
        affine = np.eye(4, dtype='float32')
        affine.ravel()[:12] = meta['affine']
        meta['affine'] = affine

    # --- Initialize chunks
    if chunks is None:
        chunks = tuple([1, data.shape[1], data.shape[2], data.shape[3]])

    # --- Initialize compression
    if compression is not None:
        assert compression in ['gzip', 'lzf'], 'Error specified compression type %s is not supported' % compression

    return fname , data, meta, chunks, compression

def load(fname, infos=None):
    """
    Method to load full array and meta dictionary

    """
    if not os.path.exists(fname):
        return None, {} 

    with h5py.File(fname, 'r') as f:

        # --- Extract data
        data = extract_data(f, infos)

        # --- Extract meta
        meta = {'affine': f.attrs['affine']}

    return data, meta

def extract_data(f, infos):
    """
    Method to parse infos dictionary into slices

    """
    infos = check_infos(infos)

    if infos is None:
        return f['data'][:]

    # --- Create shapes / points 
    dshape = np.array(f['data'].shape[:3])
    points = np.array(infos['point']) * (dshape - 1) 
    points = np.round(points)

    # --- Create slice bounds 
    shapes = np.array(infos['shape'])
    slices = points - np.floor(shapes / 2) 
    slices = np.stack((slices, slices + shapes)).T

    # --- Create padding values
    padval = np.stack((0 - slices[:, 0], slices[:, 1] - dshape)).T
    padval = padval.clip(min=0).astype('int')
    slices[:, 0] += padval[:, 0]
    slices[:, 1] -= padval[:, 1]

    # --- Create slices
    slices = [tuple(s.astype('int')) if i > 0 else (0, d) for s, i, d in zip(slices, shapes, dshape)] 
    slices = [slice(s[0], s[1]) for s in slices]

    data = f['data'][slices[0], slices[1], slices[2]]

    # --- Pad array if needed
    if padval.any():
        pad_width = [(b, a) for b, a in zip(padval[:, 0], padval[:, 1])] + [(0, 0)]
        data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=np.min(data))

    return data

def check_infos(infos):

    if infos is not None:

        assert type(infos) is dict
        assert 'point' in infos
        assert 'shape' in infos
        assert len(infos['point']) == 3
        assert len(infos['shape']) == 3

    return infos

def load_meta(fname):
    """
    Method to load meta dictionary containing:
    
      (1) shape
      (2) affine matrix

    """
    if not os.path.exists(fname):
        return {} 

    with h5py.File(fname, 'r') as f:

        meta = {
            'shape': f['data'].shape,
            'affine': f.attrs['affine']}

    return meta
