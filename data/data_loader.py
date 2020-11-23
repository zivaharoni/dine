
from data.synthetic_loader import load_synthetic_ds

def load_data(config):
    if config.data_name == "synthetic":
        return load_synthetic_ds(config)
    else:
        raise ValueError("'{}' is invalid data name".format(config.data_name))

