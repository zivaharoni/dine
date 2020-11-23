import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
import logging

logger = logging.getLogger("logger")

class ConstantScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr

    def __call__(self, step):
        return self.lr

    def get_config(self):
        pass
