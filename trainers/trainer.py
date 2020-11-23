import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.metrics import SparseCategoricalCrossentropy, SparseCategoricalAccuracy, MeanSquaredError
from tensorflow.keras.optimizers import SGD
import logging
import os
from optimizers.lr_schedulers import ConstantScheduler
from losses.information_loss import DVContinuousLoss
from metrics.metrics import CustomMetrics, DI, DVContinuous
from visualizers.visualizer import Visualization, Plot
import backend.loss_metric_utils as mK

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
logger = logging.getLogger("logger")


def build_trainer(model, data, config):
    if config.trainer_name == "di_cont":
        trainer = DINETrainer(model, data, config)
    else:
        raise ValueError("'{}' is an invalid trainer name")

    return trainer

class DINETrainer(object):
    def __init__(self, model, data, config):
        self.loss_fn = DVContinuousLoss(name="nll_loss")
        self.model = model
        self.param_num = tf.cast(model['train'].count_params(), tf.float64)
        self.data = data

        self.config = config
        self.learning_rate = ConstantScheduler(config.learning_rate)
        self.optimizer = SGD(config.learning_rate)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        def build_metrics():
            metric_pool = list()
            metric_pool.append(dict({"metric": DVContinuous("D_xy"),
                                     "name": "mi",
                                     "weight_fn": mK.nan_mask,
                                     "target_fn": mK.split_identity_0,
                                     "pred_fn": mK.split_identity_0}))
            metric_pool.append(dict({"metric": DVContinuous("D_y"),
                                     "name": "mi",
                                     "weight_fn": mK.nan_mask,
                                     "target_fn": mK.split_identity_1,
                                     "pred_fn": mK.split_identity_1}))
            metric_pool.append(dict({"metric": DI("di"),
                                     "name": "mi",
                                     "weight_fn": mK.nan_mask,
                                     "target_fn": mK.identity,
                                     "pred_fn": mK.identity}))
            return metric_pool


        self.metrics = {"train": CustomMetrics(build_metrics(),
                                               config.train_writer,
                                               name='train'),
                        "eval": CustomMetrics(build_metrics(),
                                              config.test_writer,
                                              name='eval')}

        def build_figures():
            figure_pool = dict()
            figure_pool["di"] = Plot(name='di_progress')
            return figure_pool

        self.visualizer = Visualization(build_figures(),
                                        os.path.join(config.tensor_board_dir, "visual"))



    @staticmethod
    def metrics_data(t, t_):
        return t, t_

    @staticmethod
    def visualize_aggr_data(t, t_):
        return None

    @staticmethod
    def visualize_plot_data(*args):
        visual_data = dict({"di": args[-1]})
        return visual_data

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)

    def sync_eval_model(self):
        w = self.model['train'].get_weights()
        self.model['eval'].set_weights(w)

    # @tf.function
    def compute_grads(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            T, T_ = self.model['train']([x, y], training=True)
            loss = self.loss_fn(T, T_)

            loss_xy, loss_y = tf.split(loss, num_or_size_splits=2)

        gradients_xy = tape.gradient(loss_xy, self.model['train'].trainable_weights)
        gradients_y = tape.gradient(loss_y, self.model['train'].trainable_weights)

        gradients = [g_xy + g_y for g_xy,g_y in zip(gradients_xy, gradients_y)]
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():
            tf.summary.scalar("loss_xy", tf.squeeze(loss_xy), self.global_step)
            tf.summary.scalar("loss_y", tf.squeeze(loss_y), self.global_step)
            tf.summary.scalar("loss", tf.squeeze(loss_xy) - tf.squeeze(loss_y), self.global_step)
            tf.summary.scalar("grad_norm", grad_norm, self.global_step)
            tf.summary.scalar("lr", self.learning_rate(self.global_step), self.global_step)
            self.global_step.assign_add(1)

        return gradients, T, T_

    # @tf.function
    def apply_grads(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model['train'].trainable_weights))

    # @tf.function
    def train_step(self, x, y):
        gradients, T, T_ = self.compute_grads(x, y)
        self.apply_grads(gradients)
        return T, T_

    def eval_step(self, x):
        T, T_ = self.model['eval'](x, training=False)
        return T, T_

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()

        for x, y in self.data["train"]:
            if x is None:
                self.reset_model_states()
                continue

            T, T_  = self.train_step(x, y)
            self.metrics["train"].update_state(T, T_)
        self.metrics["train"].log_metrics(epoch)

    def evaluate(self, epoch, iterator="eval"):
        self.metrics["eval"].reset_states()
        self.visualizer.reset_states()
        self.sync_eval_model()
        self.reset_model_states()

        for k,(x, y) in enumerate(self.data[iterator]):
            T, T_ = self.eval_step([x, y])
            self.metrics["eval"].update_state(*self.metrics_data(T, T_))
            self.visualizer.update_state(self.visualize_aggr_data(T, T_))

        results = self.metrics["eval"].log_metrics(epoch)
        self.visualizer.update_state(self.visualize_plot_data(*results))
        self.visualizer.plot(epoch, save=True)

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)
            self.train_epoch(epoch)
        self.evaluate(self.config.num_epochs, iterator="long_eval")