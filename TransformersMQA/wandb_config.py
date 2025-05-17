import wandb
import os

def init_wandb(project_name="transformers_MQA_100"):
    wandb.init(
        project=project_name,
        entity=os.getenv("WANDB_ENTITY"),  # Use an environment variable
        config={
            "architecture": "transformer_MQA",
            "dataset": "investment_dataset",
            "batch_size": 128,
            "epochs": 30,
            "learning_rate": 1e-3,
            "optimizer": "Adam",
            "loss_function": "CrossEntropy",
            "dropout_rate": 0.1  # fixed typo
        }
    )
    return wandb.init