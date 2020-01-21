import os, glob, importlib
from types import ModuleType

def load_modules(repo='dl_train', submodule='datasets', allows=[]):
    """
    Method to compile submodule from all available libraries

    """
    ROOT = os.environ.get('{}_ROOT'.format(repo).upper(), '.')

    is_hidden = lambda x : x.startswith('_')
    is_module = lambda x, module : isinstance(getattr(module, x), ModuleType) and x not in allows 

    inits = glob.glob('{}/{}/*/{}/__init__.py'.format(ROOT, repo, submodule))
    inits = [i.replace(ROOT, '') for i in inits]

    for init in inits:

        # --- Get path to module
        path = '.'.join(init.split('/')[1:-1])

        # --- Import module
        module = importlib.import_module(path)
        names = [n for n in module.__dict__ if not is_hidden(n) and not is_module(n, module)]
        globals().update({n: getattr(module, n) for n in names})

# --- Load modules
load_modules()

# --- Remove imports from namespace
for name in ['os', 'glob', 'importlib', 'ModuleType', 'load_modules', 'name']:
    globals().pop(name)
