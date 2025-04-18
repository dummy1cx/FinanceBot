## Initializes wandb api key = "deleted"

## --------------------------------------------------------------------------------
## This configuration is highly referenced from Sequence to Sequence Learning
## with Neural Networks by Sutskever et all 2014
## The original research was on NMT but we have tried to implement the same idea
## on chatbot training and noticed a substantial improvement in performance
## ----------------------------------------------------------------------------------
import wandb

def init_wandb(project_name="seq2seqLSTM", config=None):
    if config is None:
        config = {
            "model_name": "deep_lstm_seq2seq",
            "attn_model": "dot",
            "embedding": "glove.6B.300d",
            "embedding_dim": 300,  
            "freeze_embeddings": False,
            "hidden_size": 1000,  
            "encoder_n_layers": 4,
            "decoder_n_layers": 4,
            "dropout": 0.1,
            "batch_size": 128, 
            "learning_rate": 0.0001, 
            "decoder_learning_ratio": 1.0,  
            "teacher_forcing_ratio": 1.0,
            "clip": 5.0,  
            "n_iteration": 10000,  
            "print_every": 20,
            "save_every": 1000
        }

    wandb.init(project=project_name, config=config)
    return wandb.config


