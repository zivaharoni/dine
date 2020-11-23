import tensorflow as tf
from tensorflow.keras import backend as K

class DVContinuousLoss(tf.keras.losses.Loss):
    def __init__(self, name='dv_loss'):
        super(DVContinuousLoss, self).__init__(name=name, reduction='none')

    def call(self, T, T_, **kwargs):
        txy, ty = tf.split(T, 2, axis=-2)
        txy_, ty_ = tf.split(T_, 2, axis=-2)

        N = tf.cast(tf.reduce_prod(txy.shape[:-1]), tf.float64)
        N_ref = tf.cast(tf.reduce_prod(txy_.shape[:-1]), tf.float64)
        mean_xy_t = K.sum(txy / N)
        mean_xy_et =  tf.math.reduce_logsumexp(txy_-K.log(N_ref))
        mean_y_t = K.sum(ty / N)
        mean_y_et =  tf.math.reduce_logsumexp(ty_-K.log(N_ref))

        return tf.stack([-(mean_xy_t - mean_xy_et), -(mean_y_t - mean_y_et)])
