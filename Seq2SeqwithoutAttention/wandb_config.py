# wandb api "deleted"
#
#
import wandb


def init_wandb(project_name="finance-chatbot"):
    wandb.init(
        project=project_name,
        config={
            "batch_size": 16,
            "hidden_dim": 32,
            "embedding_dim": 50,
            "epochs": 20,
            "teacher_forcing_ratio": 0.5,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_function": "NLLLoss",
        }
    )
    return wandb.config


