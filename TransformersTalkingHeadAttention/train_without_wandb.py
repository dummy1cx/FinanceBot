from TransformersMultiHeadAttention.transformer import Transformer
import numpy as np
from torch import nn
from torch.optim import Adam
import torch
import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.utils.data import random_split, DataLoader, Subset

from TransformersMultiHeadAttention.dataset import  datasetLoader
from wandb_config import init_wandb
import  pandas as pd

# Define device at the top
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")


def train_step(src , trg) :

    decoder_input = trg[: , :-1]

    trg_reals = trg[: , 1:].reshape(-1)

    preds = model(src , decoder_input)

    preds = preds.reshape(-1 , preds.shape[2])

    optimizer.zero_grad()

    loss = criterion(preds , trg_reals)

    loss.backward()

    # Avoid exploding gradient issues
    torch.nn.utils.clip_grad_norm_(model.parameters() , max_norm=1)

    optimizer.step()

    return loss


if __name__ == "__main__":
    path = "./Cleaned_date.json"
    loader = datasetLoader(path)
    dataframe_dataloader, src_vocab_size, trg_vocab_size, src_max_len, trg_max_len, src_tokenizer, src_tokenizer = loader.define_dataloader()

    epochs = 30
    lr = 1e-3  # Learning rate
    model_dimension = 256
    inner_layer_dimension = 512
    num_layers = 4
    num_heads = 8
    dropout_rate = 0.1



    model = Transformer(num_layers,model_dimension,num_heads,inner_layer_dimension,src_vocab_size,trg_vocab_size,src_max_len, trg_max_len,dropout_rate).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.word_index['<pad>'])
    optimizer = Adam(model.parameters(), lr=lr)



    # Split dataset into training and validation
    dataset_size = len(dataframe_dataloader.dataset)  # Get the size of the dataset
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = dataset_size - train_size  # 20% for validation

    # Split the dataset indices
    indices = list(range(dataset_size))
    train_indices, val_indices = random_split(indices, [train_size, val_size])

    # Create Subset instances for training and validation
    train_dataset = Subset(dataframe_dataloader.dataset, train_indices)
    val_dataset = Subset(dataframe_dataloader.dataset, val_indices)

    # Create DataLoaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)



    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        epoch_train_loss = 0
        epoch_val_loss = 0

        # Train phase
        model.train()
        for src, trg in train_dataloader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            loss = train_step(src, trg)
            epoch_train_loss += loss

        # Calculate average training loss and train perplexity
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss.cpu().detach().numpy())
        train_perplexity = np.exp(avg_train_loss.cpu().item())  # Calculate training perplexity

        # Validation phase
        model.eval()
        with torch.no_grad():
            for src, trg in val_dataloader:
                src, trg = src.to(DEVICE), trg.to(DEVICE)

                # Get model predictions
                decoder_input = trg[:, :-1]
                trg_reals = trg[:, 1:].reshape(-1)
                preds = model(src, decoder_input)
                preds = preds.reshape(-1, preds.shape[2])

                # Calculate validation loss
                loss_val = criterion(preds, trg_reals)
                epoch_val_loss += loss_val

        # Calculate average validation loss and validation perplexity
        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss.cpu().detach().numpy()) # Store validation loss
        val_perplexity = np.exp(avg_val_loss.cpu().item())  # Calculate validation perplexity


        # Logging
        if (epoch + 1) % 5 == 0:
            print(f"\n[Epoch: {epoch+1}/{epochs}] "
                  f"[Train Loss: {train_losses[-1]:0.2f}] "
                  f"[Train Perplexity: {train_perplexity:0.2f}] \n")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'train_loss': avg_train_loss,  # Use the average train loss
            'train_perplexity': train_perplexity,
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch + 1}.pt")

    model_save_path = "./transformer_model_MQA.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

