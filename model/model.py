from model.ae import AE
from model.vae import VAE
from model.ra import RA
from model.ganomaly.lightning_model import Ganomaly


def get_model(config):
    print(f"Loading model {config['model_name']}")
    if config["model_name"] == "AE":
        return AE(config)
    elif config["model_name"] == "VAE":
        return VAE(config)
    elif config["model_name"] == "RA":
        return RA(config)
    elif config["model_name"] == "ganomaly":
        return Ganomaly(
            latent_vec_size=config["latent_vec_size"],
            batch_size=config["batch_size"],
            input_size=config["target_size"],
            n_features=config["n_features"],
            lr=config["lr"],
            extra_layers=config["extra_layers"],
            wadv=config["wadv"],
            wcon=config["wcon"],
            wenc=config["wenc"],
        )
    else:
        raise ValueError(f"Unknown model name {config['model_name']}")
