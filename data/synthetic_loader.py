import tensorflow as tf
from data.data_iterators import FixedLengthIterator

def load_synthetic_ds(config):

    if config.data_subname == "awgn":
        data_input_shape = (config.x_dim,)
        data_output_shape = (config.y_dim,)
        batch_size = config.batch_size
        lambda_fn_x = None
        lambda_fn_y = None
        gen = AWGNGenerator(snr=config.snr, input_shape=data_input_shape, dtype=tf.float64)
        iterator = FixedLengthIterator
    else:
        raise ValueError("data subname is nor implemented")


    gen_output_shapes = (data_input_shape, data_output_shape)
    gen_output_dtypes = (tf.float64, tf.float64)

    ds_train = make_dataset_from_generator(gen_fn=gen,
                                           output_shape=gen_output_shapes,
                                           output_dtype=gen_output_dtypes,
                                           batch_size=batch_size,
                                           lambda_fn_x=lambda_fn_x,
                                           lambda_fn_y=lambda_fn_y)
    train_iterator = iterator(ds_train, config.train_epoch_len)

    ds_eval = make_dataset_from_generator(gen_fn=gen,
                                          output_shape=gen_output_shapes,
                                          output_dtype=gen_output_dtypes,
                                          batch_size=batch_size,
                                          lambda_fn_x=lambda_fn_x,
                                          lambda_fn_y=lambda_fn_y)
    eval_iterator = iterator(ds_eval, config.eval_epoch_len)
    long_eval_iterator = iterator(ds_eval, config.mc_evaluation_len)

    data = {'train': train_iterator,
            'eval': eval_iterator,
            'long_eval': long_eval_iterator}

    return data

def make_dataset_from_generator(gen_fn, output_shape, output_dtype, batch_size=10, lambda_fn_x=None, lambda_fn_y=None):

    def map_transform(x,y):
        return lambda_fn_x(x), lambda_fn_y(y)

    ds = tf.data.Dataset.from_generator(gen_fn,
                                        output_shapes=output_shape,
                                        output_types=output_dtype)
    ds = ds.repeat().batch(batch_size)

    if lambda_fn_x is not None and lambda_fn_y is not None:
        ds = ds.map(map_transform)
    return ds

class AWGNGenerator(object):
    def __init__(self, snr=1., input_shape=(1,), dtype=tf.float64):
        self.SNR = tf.cast(snr, dtype)
        self.shape = input_shape
        self.dtype = dtype

    def __iter__(self):
        x = tf.random.normal(shape=self.shape, dtype=self.dtype)
        y = x + tf.sqrt(1/self.SNR) * tf.random.normal(shape=self.shape, dtype=self.dtype)
        yield x,y

    def __call__(self):
         return self
