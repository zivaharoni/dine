from models.continuous_dv_models import log_likelihood_ratio_memory_network

def build_model(config):
    if config.model_name == "cont_info":
        model = cont_info_model(config)
    else:
        raise ValueError("'{}' is an invalid model name")

    return model


def cont_info_model(config):
    if config.model_subname == "di":
        model_train = log_likelihood_ratio_memory_network(config)
        model_eval = log_likelihood_ratio_memory_network(config)
    else:
        raise ValueError("'{}' is an invalid model sub-name")

    return {"train": model_train, "eval": model_eval}

