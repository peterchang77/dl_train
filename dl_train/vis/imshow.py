import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def prepare_array(x):
    """
    Method to ensure the array is a properly formatted 2D matrix 

    """
    if x is None:
        return x

    x = np.squeeze(x.copy())

    if x.ndim == 3:
        x = x[..., 0]

    assert x.ndim == 2, 'ERROR input must be in H x W (x C) format'

    return x

def imshow(dat, lbl=None, radius=1, vm=None, title=None, figsize=(7, 7)):
    """
    Method to display dat with lbl overlay if provided

    :params

      (np.array) dat : 2D dat array of format H x W or H x W x C
      (np.array) lbl : 2D lbl array of format H x W or H x W x N

      (int) radius : thickness of outline for overlays 
      (int) vm[0] : lower range of visualized values
      (int) vm[1] : upper range of visualized values

    """
    x = prepare_array(dat)
    m = prepare_array(lbl.astype('uint8'))

    # --- Overlay if lbl also provided
    if m is not None:
        if m.max() > 0:

            perim = lambda msk, radius : ndimage.binary_dilation(msk, iterations=radius) ^ (msk > 0)
            masks = [perim(m == c, radius) for c in range(m.max())]
            masks = np.stack(masks, axis=-1)
            x = imoverlay(x, masks) 

    # --- Display image
    plt.figure(figsize=figsize)
    plt.axis('off')

    kwargs = {}
    kwargs['cmap'] = plt.cm.gist_gray
    kwargs['vmin'] = None if vm is None else vm[0]
    kwargs['vmax'] = None if vm is None else vm[1]

    plt.imshow(x, **kwargs)

    if title is not None:
        plt.title(title)

def imoverlay(dat, lbl, vm=None):
    """
    Method to superimpose lbl on 2D image

    :params

      (np.array) dat : 2D dat array of format H x W or H x W x C
      (np.array) lbl : 2D lbl array of format H x W or H x W x N

    """
    dat = prepare_array(dat)
    lbl = prepare_array(lbl)

    # --- Prepare dat 
    if dat.ndim  == 2:
        dat = gray_to_rgb(dat, vm=vm)

    # --- Prepare lbl 
    if lbl.ndim  == 2:
        lbl = np.expand_dims(lbl, axis=2)

    lbl = lbl.astype('bool')

    # --- Overlay lbl(s)
    rgb = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    overlay = []

    for channel in range(3):
        layer = dat[..., channel]

        for i in range(lbl.shape[2]):
            layer[lbl[..., i]] = rgb[i % 6][channel]

        overlay.append(layer)

    return np.stack(overlay, axis=-1)

def gray_to_rgb(dat, max_val=1, percentile=0, vm=None):
    """
    Method to convert H x W grayscale array to H x W x 3 RGB grayscale

    :params

    (int) max_val : maximum value in output array

      if max_val == 1, output is assumed to be float32
      if max_val >  1, output is assumed to be uint8 (RGB)

    (int) percentile : lower bound to set to 0

    """
    if vm is None:
        vmin, vmax = np.percentile(dat, percentile), np.percentile(dat, 100 - percentile)
    else:
        vmin, vmax = vm[0], vm[1]

    den = vmax - vmin 
    den = 1 if den == 0 else den
    dat = ((dat - vmin) / den).clip(min=0, max=1) * max_val
    dat = np.stack([dat] * 3, axis=-1)

    dtype = 'float32' if max_val == 1 else 'uint8'

    return dat.astype(dtype)

