import tensorflow as tf
from tensorflow.keras import losses, metrics, backend

# =================================================================
# CUSTOM KERAS LOSSES + METRICS
# =================================================================

def sce(weights, scale=1.0):

    loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    def sce(y_true, y_pred):

        return loss(y_true, y_pred, weights) * scale

    return sce 

def mse(weights, scale=1.0):

    loss = losses.MeanSquaredError()

    def mse(y_true, y_pred):

        return loss(y_true, y_pred, weights) * scale

    return mse

def mae(weights, scale=1.0):

    loss = losses.MeanAbsoluteError()

    def mae(y_true, y_pred):

        return loss(y_true, y_pred, weights) * scale

    return mae

def sl1(weights, scale=1.0, delta=1.0):

    loss = losses.Huber(delta=delta)

    def sl1(y_true, y_pred):

        return loss(y_true, y_pred, weights) * scale

    return sl1

def dsc(weights=None, scale=1.0, epsilon=1):

    def dice(y_true, y_pred):

        true = y_true[..., 0]  == 1
        pred = y_pred[..., 1] > y_pred[..., 0] 

        if weights is not None:
            true = true & (weights[..., 0] != 0) 
            pred = pred & (weights[..., 0] != 0)

        A = tf.math.count_nonzero(true & pred) * 2
        B = tf.math.count_nonzero(true) + tf.math.count_nonzero(pred) + epsilon

        return (A / B) * scale

    return dice

def acc(weights):

    metric = metrics.Accuracy()

    def accuracy(y_true, y_pred):

        true = y_true[..., 0]
        pred = backend.argmax(y_pred)

        return metric(true, pred, weights)

    return accuracy

# =================================================================
# CUSTOM LAYERS AND FUNCTIONS 
# =================================================================

def flatten(x):
    """
    Method to flatten all defined axes (e.g. not None)

    WARNING: If possible, layers.Flatten(...) is preferred for speed and HDF5 serialization compatibility

    """
    # --- Calculate shape
    ll = x._shape_as_list()
    ss = [s for s in tf.shape(x)]

    shape = []
    adims = []

    for l, s in zip(ll, ss):
        if l is None:
            shape.append(s)
        else:
            shape.append(1)
            adims.append(l)

    shape[-1] = np.prod(adims)

    return tf.reshape(x, shape)

