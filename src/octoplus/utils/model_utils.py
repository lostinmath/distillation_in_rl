import pickle


def get_model(config):
    if config.path_to_weights:
        model = load_model(config.path_to_weights)
    else:
        from octo.model.octo_model import OctoModel

        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    return model


def load_model(
    path_to_weights: str = "data/model_weights/octo_small_weights.pkl",
) -> object:
    """Load a model from a pickle file.

    Args:
        path_to_weights (str): path to the pickle file containing the model.

    Returns:
        model: the model loaded from the pickle file.
    """
    pkl_file = open(path_to_weights, "rb")
    model = pickle.load(pkl_file)
    pkl_file.close()
    return model


def save_model(model: object, path_to_weights: str = "") -> None:
    """Save a model to a pickle file.

    Args:
        model (object): the model to save.
        path_to_weights (str): path to the pickle file where the model will be saved.
    """
    output = open(path_to_weights, "wb")
    pickle.dump(model, output)
    output.close()


def load_octo_from_api():
    """
    Loads Octo using the api provided in Octo repo
    """
    from octo.model.octo_model import OctoModel

    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    # from transformers import AutoModel
    # model = AutoModel.from_pretrained("rail-berkeley/octo-base-1.5")
    return model


def load_octo_from_local():
    """
    Loads Octo from the local pickle file. Faster and more stable.
    """
    from octoplus.src.utils.model_utils import load_model

    model = load_model("octoplus/data/model_weights/octo_small_weights.pkl")
