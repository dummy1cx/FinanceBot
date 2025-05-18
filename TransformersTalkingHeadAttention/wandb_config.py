import wandb
import os

def init_wandb(project_name="transformers_encoder_decoder_talking_head_90"):
    wandb.init(
        project=project_name,
        entity=os.getenv("WANDB_ENTITY"),  # Use an environment variable
        config={
            "architecture": "encoder_decoder_transformers_talking_head",
            "dataset": "investment_dataset",
            "batch_size": 128,
            "epochs": 80,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "loss_function": "CrossEntropy",
            "dropout_rate": 0.1
        }
    )
    return wandb.init