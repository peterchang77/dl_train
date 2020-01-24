import numpy as np, os
import types, importlib
from tensorflow.keras import models
from dl_utils.general import *

class Trained():

    def __init__(self, path):

        self.model = None

        self.jspecs = {}
        self.kwargs = {
            'predict': {'softmax': False},
            'pre': {},
            'post': {}}

        self.load_py(path)
        self.load_h5(path)

    def load_h5(self, path, compile=False):

        h5 = '{}/model.hdf5'.format(path)

        if os.path.exists(h5):
            self.model = models.load_model(h5, compile=compile)

    def load_py(self, path):

        py = '{}/data.py'.format(path)

        if os.path.exists(py):

            loader = importlib.machinery.SourceFileLoader('data', py)
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)

            # --- Grab custom code
            for func in ['preprocess', 'postprocess']:
                if hasattr(mod, func):
                    setattr(self, func, getattr(mod, func))

            # --- Grab custom attr
            for attr in ['kwargs', 'jspecs']:
                if hasattr(mod, attr):
                    getattr(self, attr).update(getattr(mod, attr))

    def predict(self, xs, softmax=False, reduce_dims=True, **kwargs): 

        if self.model is None:
            printd('ERROR model has not yet been loaded')
            return

        # --- Prepare xs 
        for k in xs:
            assert k in self.model.input
            if xs[k].ndim == 4:
                xs[k] = np.expand_dims(xs[k], axis=0)

        # --- Create fillers remaining keys
        for k, v in self.model.input.items():
            if k not in xs:
                shape = [s if s is not None else 1 for s in v.shape.as_list()]
                dtype = v.dtype.name
                xs[k] = np.ones(shape, dtype=dtype)

        # --- Convert to softmax
        if type(softmax) is bool:
            softmax = {k : softmax for k in self.model.output_names}
        softmax = [softmax[k] for k in self.model.output_names]

        ys = self.model.predict(xs)

        if reduce_dims:
            ys = [y[0] if y.ndim == 5 else y for y in ys]

        return {k: self.softmax(y) if s else y for k, y, s in zip(self.model.output_names, ys, softmax)}

    def softmax(self, arr):

        return np.exp(arr) / np.sum(np.exp(arr), axis=-1, keepdims=True)

    def run(self, xs, **kwargs):

        if type(xs) is not dict:
            xs = {'dat': xs}

        xs = self.preprocess(xs, **{**self.kwargs['pre'], **kwargs})
        ys = self.predict(xs, **{**self.kwargs['predict'], **kwargs})
        ys = self.postprocess(ys, **{**self.kwargs['post'], **kwargs})

        return ys

    def preprocess(self, xs, **kwargs):

        return xs

    def postprocess(self, ys, **kwargs):

        return ys