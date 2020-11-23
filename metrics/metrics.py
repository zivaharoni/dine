import pickle
import numpy as np
import scipy
import os
import time
from datetime import datetime
from collections import defaultdict as def_dict
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric, Mean, SparseCategoricalAccuracy, SparseCategoricalCrossentropy
import logging
from backend.loss_metric_utils import nan_mask, nan_zero_mask, nan_not_zero_mask, identity
logger = logging.getLogger("logger")


class ModelMetrics(Metric):

    def __init__(self, writer, name='', **kwargs):
        super(ModelMetrics, self).__init__(name=name, **kwargs)
        self.writer = writer
        self.metric_pool = list()
        self.names = list()
        self.weight_fn = list()
        self.target_fn = list()
        self.pred_fn = list()
        self.t_start = time.time()

    def update_state(self, targets, prediction, sample_weight=None):
        for metric, name, weight_fn, pred_fn, target_fn  in zip(self.metric_pool, self.names, self.weight_fn, self.pred_fn, self.target_fn):
            if isinstance(targets, dict):
                tar = targets[name]
                pred = prediction[name]
            else:
                tar = targets
                pred = prediction

            metric.update_state(target_fn(tar), pred_fn(pred), sample_weight=weight_fn(tar))

    def result(self):
        return [metric.result() for metric in self.metric_pool]

    def reset_states(self):
        self.t_start = time.time()
        for metric in self.metric_pool:
            metric.reset_states()

    def log_metrics(self, epoch):
        # log to tensorboard
        with self.writer.as_default():
            for metric in self.metric_pool:
                tf.summary.scalar(metric.name, metric.result(), epoch)

        # print to terminal
        msg = ["{:8s} {:8d} | {:8} {:8.2f}".format(self.name, epoch, "time", time.time() - self.t_start)]
        for metric in self.metric_pool:
            if np.isnan(metric.result()):
                logger.info("NaN appeared in metric {}".format(metric.name))
            msg.append("{:8s} {: 8.6f}".format(metric.name, float(metric.result())))
        if "eval" in self.name:
            logger.info("-" * len("   ".join(msg)))
        logger.info(" | ".join(msg))
        if "eval" in self.name:
            logger.info("-" * len("   ".join(msg)))

        return self.result()

class CustomMetrics(ModelMetrics):

    def __init__(self, metric_pool, writer, name='', **kwargs):
        super(CustomMetrics, self).__init__(writer, name=name, **kwargs)

        for tup in metric_pool:
            self.metric_pool.append(tup["metric"])
            self.names.append(tup["name"])
            self.weight_fn.append(tup["weight_fn"])
            self.target_fn.append(tup["target_fn"])
            self.pred_fn.append(tup["pred_fn"])

##################################
class DI(Metric):
    def __init__(self, name='dv_loss', **kwargs):
        super(DI, self).__init__(name=name, **kwargs)
        self.txy = self.add_weight(name='txy', initializer='zeros')
        self.exp_txy_bar = self.add_weight(name='exp_txy_bar', initializer='zeros')
        self.ty = self.add_weight(name='ty', initializer='zeros')
        self.exp_ty_bar = self.add_weight(name='exp_ty_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')


    def update_state(self, T, T_, **kwargs):
        txy, ty = tf.split(T, 2, axis=-2)
        txy_, ty_ = tf.split(T_, 2, axis=-2)
        shape = tf.cast(tf.shape(ty), tf.float64)
        shape_ = tf.cast(tf.shape(ty_), tf.float64)

        self.txy.assign(self.txy +  tf.reduce_sum(txy))
        self.exp_txy_bar.assign(self.exp_txy_bar + tf.reduce_sum(K.exp(txy_)))

        self.ty.assign(self.ty +  tf.reduce_sum(ty))
        self.exp_ty_bar.assign(self.exp_ty_bar + tf.reduce_sum(K.exp(ty_)))

        self.global_counter.assign(self.global_counter + tf.reduce_prod(shape[:-1]))
        self.global_counter_ref.assign(self.global_counter_ref + tf.reduce_prod(shape_[:-1]))

    def result(self):
        loss_xy = self.txy / self.global_counter - K.log(self.exp_txy_bar / self.global_counter_ref)
        loss_y = self.ty / self.global_counter - K.log(self.exp_ty_bar / self.global_counter_ref)
        return loss_xy - loss_y

class DVContinuous(Metric):
    def __init__(self, name='dv_loss', **kwargs):
        super(DVContinuous, self).__init__(name=name, **kwargs)
        self.t = self.add_weight(name='t', initializer='zeros')
        self.exp_t_bar = self.add_weight(name='exp_t_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')

    def update_state(self, t, t_, **kwargs):

        self.t.assign(self.t + tf.reduce_sum(t))
        self.exp_t_bar.assign(self.exp_t_bar + tf.reduce_sum(K.exp(t_)))

        self.global_counter.assign(self.global_counter + tf.cast(tf.reduce_prod(t.shape[:-1]), tf.float64))
        self.global_counter_ref.assign(self.global_counter_ref + tf.cast(tf.reduce_prod(t_.shape[:-1]), tf.float64))

    def result(self):
        loss = self.t / self.global_counter - K.log(self.exp_t_bar / self.global_counter_ref)
        return loss
