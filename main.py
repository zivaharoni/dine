import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(2)
from trainers.trainer import build_trainer
from data.data_loader import load_data
from models.models import build_model
import logging
from utils.utils import preprocess_meta_data
logger = logging.getLogger("logger")

def main():
    # capture the config path from the run arguments
    # then process configuration file
    config = preprocess_meta_data()

    # load the data
    data = load_data(config)

    if not config.quiet:
        config.print()

    # create a model
    model = build_model(config)

    # create trainer and pass all the previous components to it
    trainer = build_trainer(model, data, config)

    # train the model
    trainer.train()


if __name__ == '__main__':
    main()









