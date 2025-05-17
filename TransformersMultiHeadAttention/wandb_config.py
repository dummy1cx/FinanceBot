import wandb
import os

def init_wandb(project_name="transformers_encoder_decoder_8"):
    wandb.init(
        project=project_name,
        entity=os.getenv("WANDB_ENTITY"),  # Use an environment variable
        config={
            "architecture": "encoder_decoder_transformers",
            "dataset": "investment_dataset",
            "batch_size": 128,
            "epochs": 80,
            "learning_rate": 5e-4,
            "optimizer": "Adam",
            "loss_function": "CrossEntropy",
            "dropout_rate": 0.1  # fixed typo
        }
    )
    return wandb.init