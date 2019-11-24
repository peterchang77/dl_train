import numpy as np
from tensorflow import losses, optimizers, distribute, math
from tensorflow.keras import Input, Model, layers

from client import Client

# =======================================================================
client = Client(SUMMARY_PATH='../../data/pkls/summary.pkl')
client.load_summary()
client.prepare_cohorts(fold=0)
client.set_sampling_rates(rates={
    1: 0.5,
    2: 0.5})
# =======================================================================

def generator(batch=8):

    while True:

        xs, ys = [], []

        for i in range(batch):

            x, y = client.get()
            xs.append(x)
            ys.append(y[..., 0])

        yield np.stack(xs), np.stack(ys)

def create_unet():

    inputs = Input(shape=(1, 512, 512, 1))

    # ===================================================================
    # CONTRACTING LAYERS
    # ===================================================================

    l1 = create_block(inputs, filters=[8, 8])

    STRIDES = [(1, 2, 2), (1, 1, 1)]
    l2 = create_block(l1, filters=[32, 32], strides=STRIDES)
    l3 = create_block(l2, filters=[48, 48], strides=STRIDES)
    l4 = create_block(l3, filters=[64, 64], strides=STRIDES)

    # ===================================================================
    # EXPANDING LAYERS
    # ===================================================================

    # TODO: EDIT

    CONV_LAYERS = ['conv3dt', 'conv3d']
    l5 = create_block(l4, filters=[48, 48], strides=STRIDES, conv_layers=CONV_LAYERS)
    l6 = create_block(l3 + l5, filters=[32, 32], strides=STRIDES, conv_layers=CONV_LAYERS)
    l7 = create_block(l2 + l6, filters=[8, 8], strides=STRIDES, conv_layers=CONV_LAYERS)

    # --- Final conv
    l8 = create_block(l1 + l7, filters=[8, 8], strides=STRIDES, conv_layers=CONV_LAYERS)
    outputs = layers.Conv3D(2, 7, padding='same', kernel_initializer='he_normal')(l1 + l7) 

    model = Model(inputs=inputs, outputs=outputs) 

    return model

def create_block(x, filters, strides, **kwargs):

    ACTIVATIONS = {
        'relu': layers.ReLU,
        'leaky': layers.LeakyReLU,
        'none': lambda **kwargs : lambda x : x}

    CONV_LAYERS = {
        'conv2d': layers.Conv2D,
        'conv3d': layers.Conv3D,
        'conv2dt': layers.Conv2DTranspose,
        'conv3dt': layers.Conv3DTranspose,
        'none': lambda **kwargs : lambda x : x}

    assert type(filters) is list
    assert type(strides) is list

    # --- Create arg dicts
    kwargs_ = {
        'kernel_size': 3,
        'activations': 'leaky',
        'conv_layers': 'conv3d',
        'padding': 'same',
        'kernel_initializer': 'he_normal'}

    kwargs_.update(kwargs)

    for k in kwargs_:
        if type(kwargs_[k]) in [int, str]: 
            kwargs_[k] = [kwargs_[k]] * len(filters)

    kwargs_['filters'] = filters
    kwargs_['strides'] = strides 

    extract = lambda d, index : {k: v[index] for k, v in d.items()}
    kwargs_ = [extract(kwargs_, i) for i in range(len(filters))]

    for k in kwargs_:

        conv_layer = k.pop('conv_layers')
        activation = k.pop('activations')

        assert conv_layer in CONV_LAYERS
        assert activation in ACTIVATIONS

        # --- Create convolutional layer
        x = CONV_LAYERS[conv_layer](**k)(x)

        # --- Create batch norm layer
        x = layers.BatchNormalization()(x)

        # --- Create activation layer
        x = ACTIVATIONS[activation]()(x)

    return x

def metric_dice():

    def dice(y_true, y_pred):

        true = y_true == 1
        pred = y_pred[..., 1] > y_pred[..., 0] 

        A = math.count_nonzero(true & pred) * 2
        B = math.count_nonzero(true) + math.count_nonzero(pred)

        return A / B

    return dice

def prepare_model():

    model = create_unet()
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    dice = metric_dice()
    optimizer = optimizers.Adam(learning_rate=2e-4)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', dice])

    return model

# =================================================================
#
# mirrored_strategy = distribute.MirroredStrategy()
# with mirrored_strategy.scope():
#     model = prepare_model()
#     model.fit(generator())
#
# =================================================================
#
# model = prepare_model()
# model.fit_generator(
#     generator=generator(),
#     steps_per_epoch=100,
#     epochs=2)
#
# =================================================================
#
# g = generator()
# model = prepare_model()
# count = 0
# for x, y in g:
#     result = model.train_on_batch(x, y)
#     print('iteration %05i - loss: %0.4f' % (count, result[0]), end='\r')
#     count += 1
#
# =================================================================
