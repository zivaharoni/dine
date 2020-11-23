import datetime
import logging
import os
import tensorflow as tf
logger = logging.getLogger("logger")

def set_logger(config):
    ''' define logger object to log into file and to stdout'''

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(message)s")

    if not config.quiet:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    log_path = os.path.join(config.tensor_board_dir, "logger.log")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    logger.propagate = False

def set_logger_and_tracker(config):
    '''
        1. set tracking location (uri)
        2. configure exp name/id
        3. define parameters to be documented
    '''

    dir_path = list()
    dir_path.append('.')
    dir_path.append('results')
    dir_path.append(config.exp_name)
    dir_path.append(config.trainer_name)
    dir_path.append("{}_{}".format(config.data_name, config.data_subname))
    dir_path.append("{}_{}".format(config.model_name, config.model_subname))
    if config.run_name is not None:
        dir_path.append(config.run_name)
    if config.tag_name is not None:
        dir_path.append(config.tag_name)
    dir_path.append("{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                          config.seed))
    config.tensor_board_dir = os.path.join(*dir_path)

    if not os.path.exists(config.tensor_board_dir):
        os.makedirs(config.tensor_board_dir)

    path = os.path.join(config.tensor_board_dir, 'visual')
    if not os.path.exists(path):
        os.makedirs(path)

    train_log_dir = os.path.join(config.tensor_board_dir, 'train')
    config.train_writer = tf.summary.create_file_writer(train_log_dir)

    test_log_dir =  os.path.join(config.tensor_board_dir, 'test')
    config.test_writer = tf.summary.create_file_writer(test_log_dir)

    set_logger(config)
