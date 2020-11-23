import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Input, \
    Dropout, LSTM, Flatten, Conv2D, MaxPooling2D
from models.layers import ContrastiveNoiseLayer, ResBlock, LSTMContrastive
tf.keras.backend.set_floatx('float64')


def log_likelihood_ratio_memory_network(config):
    def process_model(input_shape, output_shape, name):
        model = Sequential(name=name)
        model.add(LSTMContrastive(output_shape, return_sequences=True, stateful=True,
                                  contrastive_duplicates=config.contrastive_duplicates,
                                  batch_input_shape=input_shape))
        return model

    def llr_mapping(name):
        m = Sequential(name=name)
        m.add(Dense(config.hidden_size, activation='tanh', kernel_initializer='he_uniform'))
        m.add(Dense(1, activation=None, kernel_constraint=keras.constraints.MaxNorm(config.max_norm)))
        return m

    contrastive_noise = ContrastiveNoiseLayer(duplicates=config.contrastive_duplicates,
                                              name="contrastive_noise")
    concat_y_y_tilde = Lambda(lambda a: tf.concat([a[0]] +
                                              [tf.squeeze(b, axis=-2)
                                               for b in tf.split(a[1],
                                                                 num_or_size_splits=config.contrastive_duplicates,
                                                                 axis=-2)],
                                              axis=-1), name="concat_y_y_tilde")
    concat_axis_m1 = Concatenate(axis=-1)
    concat_axis_m2 = Concatenate(axis=-2)
    concat_xy_ = Lambda(lambda a: tf.concat([tf.concat([b, a[1]], axis=-1)
                                             for b in tf.split(a[0], num_or_size_splits=config.contrastive_duplicates, axis=-1)],
                                            axis=-1), name="concat_xy_")
    split_E = Lambda(lambda a: tf.split(a, num_or_size_splits=[config.embed_size ,
                                                               config.contrastive_duplicates * config.embed_size ],
                                        axis=-1),
                     name="split_E")
    stack_contrastive_axis_m2 = Lambda(lambda a: tf.stack(tf.split(a, num_or_size_splits=config.contrastive_duplicates,
                                                                  axis=-1), axis=-2), name="stack_contrastive_axis_m2")
    expand_dims_axis_m2 = Lambda(lambda a: tf.expand_dims(a, axis=-2), name="expand_dims_axis_m2")

    process_y = process_model(input_shape=[config.batch_size, config.bptt, (config.contrastive_duplicates+1) * config.y_dim],
                              output_shape=config.embed_size, name="process_y")
    process_xy = process_model(input_shape=[config.batch_size, config.bptt, (config.contrastive_duplicates+1) * (config.embed_size + config.x_dim)],
                               output_shape=config.embed_size, name="process_xy")
    llr_y = llr_mapping("llr_y")
    llr_xy = llr_mapping("llr_xy")

    # model feed forward

    x = Input(shape=[config.bptt, config.x_dim], name="input_x")
    y = Input(shape=[config.bptt, config.y_dim], name="input_y")
    y_tilde = contrastive_noise(y)

    y_stacked = concat_y_y_tilde((y, y_tilde))
    E_y = process_y(y_stacked)
    e_y, e_y_ = split_E(E_y)

    xy_stacked = concat_axis_m1([concat_axis_m1([e_y, x]), concat_xy_([e_y_, x])])
    E_xy = process_xy(xy_stacked)
    e_xy, e_xy_ = split_E(E_xy)

    t_y = llr_y(expand_dims_axis_m2(e_y))
    t_y_ = llr_y(stack_contrastive_axis_m2(e_y_))

    t_xy = llr_xy(expand_dims_axis_m2(e_xy))
    t_xy_ = llr_xy(stack_contrastive_axis_m2(e_xy_))

    T = concat_axis_m2((t_xy, t_y))
    T_ = concat_axis_m2((t_xy_, t_y_))

    model = keras.models.Model(inputs=(x,y), outputs=(T, T_))
    model.summary()
    return model
