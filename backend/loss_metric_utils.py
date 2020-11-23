import tensorflow as tf
import logging

logger = logging.getLogger("logger")

# weights methods

def nan_mask(x):
    return tf.logical_not(tf.math.is_nan(tf.cast(x, tf.float64)))

def zero_mask(x):
    return tf.equal(x, tf.zeros_like(x))

def not_zero_mask(x):
    return tf.not_equal(x, tf.zeros_like(x))

def one_mask(x):
    return tf.equal(x, tf.ones_like(x))

def not_one_mask(x):
    return tf.not_equal(x, tf.ones_like(x))

def nan_zero_mask(x):
    return tf.logical_and(nan_mask(x), zero_mask(x))

def nan_not_zero_mask(x):
    return tf.logical_and(nan_mask(x), not_zero_mask(x))

def nan_one_mask(x):
    return tf.logical_and(nan_mask(x), one_mask(x))

def nan_not_one_mask(x):
    return tf.logical_and(nan_mask(x), not_one_mask(x))


# target methods
def remove_nan(x):
    return tf.where(nan_mask(x), x, tf.zeros_like(x))

def remove_nan_ones(x):
    return tf.where(nan_one_mask(x), x, tf.ones_like(x))

def tar_ce(x):
    return remove_nan(x + 1)

def tar_hinge(x):
    return remove_nan(x)


# pred methods
def identity(x):
    return x


def split_identity_0(x):
    x = tf.split(x, num_or_size_splits=2, axis=-2)
    return x[0]

def split_identity_1(x):
    x = tf.split(x, num_or_size_splits=2, axis=-2)
    return x[1]
