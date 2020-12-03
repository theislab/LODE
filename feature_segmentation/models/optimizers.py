import keras


def get_optimizer(params):
    """
    Parameters
    ----------
    params : dict; params for model run

    Returns
    -------

    """
    optimizer = None

    available_optimizers = ["Adadelta", "Adam"]
    assert params.optimizer in available_optimizers, f"optimizer not available, choose on of {available_optimizers}"

    if params.optimizer == "Adadelta":
        optimizer = keras.optimizers.Adadelta(
            learning_rate = params.learning_rate, name = "Adadelta")

    if params.optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate = params.learning_rate)
    return optimizer
