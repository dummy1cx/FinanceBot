# wandb_config.py
import wandb

def init_wandb(project_name="seq2seqRNNXAtten_last", config=None):
    """
    Initializes wandb api key = "####################deleted"
    """
    if config is None:
        config = {
            "model_name": "cb_model",
            "attn_model": "dot",
            "hidden_size": 500,
            "encoder_n_layers": 2,
            "decoder_n_layers": 2,
            "dropout": 0.1,
            "batch_size": 64,
            "learning_rate": 0.0001,
            "decoder_learning_ratio": 5.0,
            "teacher_forcing_ratio": 0.9,
            "clip": 50.0,
            "n_iteration": 4000,
            "print_every": 10,
            "save_every": 500
        }

    wandb.init(project=project_name, config=config)
    return wandb.config
