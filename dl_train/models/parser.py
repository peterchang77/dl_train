import numpy as np, pandas as pd

def from_csv(fname):

    pass

def to_csv(model, fname=None):
    """
    Method to parse Tensorflow / Keras Model() object into CSV file 

    """
    # ============================================================
    # PARSE MODEL INTO LAYER UNITS
    # ============================================================
    parsed = parse_to_layers(model)

    # ============================================================
    # EXTRACT CONFIGS FROM LAYER UNITS
    # ============================================================
    df = extract_from_layers(parsed, model)

    if fname is not None:
        df.to_csv(fname, index=False)

    return df

def parse_to_layers(model):
    """
    Method to parse Tensorflow / Keras Model() object into layer units 

    A layer unit is composed of:

      * BASE : conv, conv-transpose, flatten or dense
      * NORM : batch normalization (optional)
      * RELU : activation function (optional)

    :return

      (list) parsed : list of layer units organized as dict

        [{
          # --- layer-001
          'conv': {...},
          'norm': {...},
          'relu': {...}}, {

          # --- layer-002
          'conv': {...},
          'norm': {...},
          'relu': {...}}, {... 

          # --- layer-xxx
          'tran': {...},
          'norm': {...},
          'relu': {...}}, {...

          # --- layer-xxx
          'tran': {...}}, ...
        ]

    """
    LAYERS_BASE = ['dens', 'flat', 'tran', 'conv']
    LAYERS_REST = ['norm', 'relu']

    trainable_weights = lambda x : sum([np.prod(t.shape) for t in getattr(x, 'trainable_weights', [])])
    
    # --- Parse operations into [{BASE} - NORM - RELU] layers
    parsed = [{}]
    for layer in model.layers:
        for op in ['input', 'flat', 'dens', 'tran', 'conv', 'norm', 'relu']:
            if layer.name.replace('_', '').find(op) > -1:

                if op in LAYERS_BASE: 
                    parsed.append({})

                # --- Add all layer config information
                parsed[-1][op] = layer.get_config()

                # --- Add layer name 
                parsed[-1][op]['name'] = layer.get_output_at(0).name

                # --- Add trainable_weights
                parsed[-1][op]['trainable_weights'] = trainable_weights(layer)

                # --- Add output_shape
                if op in LAYERS_BASE:
                    parsed[-1][op]['output_shape'] = getattr(layer, 'output_shape')

                # --- Add filters from units (special case for Dense layers)
                if op in ['dens']:
                    parsed[-1][op]['kernel_size'] = (layer.input_shape[1], layer.output_shape[1]) 
                    parsed[-1][op]['filters'] = parsed[-1][op]['units']

                break

    return parsed

def extract_from_layers(parsed, model):
    """
    Method to extract detailed configuration from parsed layer units

    :return

      (pd.DataFrame) df

    """
    LAYERS_BASE = ['dens', 'flat', 'tran', 'conv']

    EXTRACT = {
        'conv': ['kernel_size', 'filters', 'strides', 'padding', 'output_shape'],
        'tran': ['kernel_size', 'filters', 'strides', 'padding', 'output_shape'],
        'dens': ['kernel_size', 'filters', 'strides', 'padding', 'output_shape'],
        'flat': ['kernel_size', 'filters', 'strides', 'padding', 'output_shape'],
        'norm': ['batch_norm'],
        'relu': ['activation']}

    PARSERS = {
        'kernel_size': lambda x : x.get('kernel_size', 'n/a'),
        'filters': lambda x : x.get('filters', 'n/a'),
        'strides': lambda x : x.get('strides', 'n/a'),
        'padding': lambda x : x.get('padding', 'n/a'),
        'output_shape': lambda x : x.get('output_shape', (None, 0, 0, 0, 0))[1:],
        'batch_norm': lambda x : True,
        'activation': lambda x : {'leaky': 'Leaky Relu', 're_lu': 'ReLU', 'undef': 'n/a'}[x.get('name', 'undef')[:5]]}

    KEYS = ['name', 'type', 'kernel_size', 'filters', 'strides', 'padding', 'output_shape', 'batch_norm', 'activation', 'residual', 'trainable_weights']
    df = {k: [] for k in KEYS}

    # --- Iterate through all layers (skip input)
    for n, layer in enumerate(parsed[1:]):

        layer_name = 'layer-%03i' % (n + 1)
        layer_type = None

        # --- Determine layer type
        for key in LAYERS_BASE:
            if key in layer:
                layer_type = key
                break

        # --- Extract all information from layer
        for op in [layer_type, 'norm', 'relu']:
            for key in EXTRACT[op]:
                df[key].append(PARSERS[key](layer[op]) if op in layer else None)

        # --- Extract layer name, type and # of trainable weights 
        df['name'].append(layer_name)
        df['type'].append(layer_type)
        df['trainable_weights'].append(sum([l['trainable_weights'] for l in layer.values()]))

    # --- Extract skip connections
    df['residual'] = extract_connections(model, parsed[1:], OP='add')

    # --- Create dataframe
    df = pd.DataFrame(df) 

    return df[KEYS]

def extract_connections(model, parsed, OP='add'):
    """
    Method to find all skip connections

    """
    # --- Find all names in all layer units
    names_in_layer = {}
    for n, layer in enumerate(parsed):
        names_in_layer['layer-%03i' % (n + 1)] = [layer[op]['name'] for op in layer]

    # --- Extract skip connections / residual operations
    connections = [None] * len(parsed)
    find_layer = lambda x : [k for k, v in names_in_layer.items() if x in v][0]

    for layer in model.layers:
        if layer.name.find(OP) > -1:
            inputs = [int(find_layer(l.name)[-3:]) for l in layer.get_input_at(0)]
            connections[max(inputs) - 1] = 'layer-%03i' % min(inputs)

    return connections
