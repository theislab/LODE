import keras


def get_optimizer(params, training_operations):
    """
    Parameters
    ----------
    params : dict; params for model run

    Returns
    -------

    """
    optimizer = None

    available_optimizers = ["Adadelta", "Adam", "SGD"]
    assert params.optimizer in available_optimizers, f"optimizer not available, choose on of {available_optimizers}"

    if params.optimizer == "Adadelta":
        optimizer = keras.optimizers.Adadelta(
            learning_rate = training_operations.callbacks_()[0])

    if params.optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate = training_operations.callbacks_()[0])

    if params.optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate = training_operations.callbacks_()[0], momentum = 0.9)
    return optimizer
