import os
import json
import pickle
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import build_vocab, collate_fn, load_json
from dataset import FinanceDataset
from model import Encoder, Decoder, Seq2Seq
from wandb_config import init_wandb

# Set environment config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize wandb for model tracking and evaluation metrics
config = init_wandb()

# configuration of device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load and reduce dataset
data = load_json('/content/Cleaned_date.json')
data = data[:int(len(data) * 0.25)]  
# use 25% of the data for faster debug
# as no text pre processign done for training the model
# the model was trained on a smaller size of dataset
# this was done to understand model behaviou and it was observed that
# the model takes more time to converge even using gpu like A100

# train-validation split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# train test file saving for further experiments
with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=2)
with open('val.json', 'w') as f:
    json.dump(val_data, f, indent=2)

# Build vocabulary for training
vocab = build_vocab(train_data)

# for inference we will need the vocab from trained dataset
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# wandb saving of vocal.pkl
vocab_artifact = wandb.Artifact("finance-vocab", type="vocab")
vocab_artifact.add_file("vocab.pkl")
wandb.log_artifact(vocab_artifact)

# dataloader for model training
train_dataset = FinanceDataset('train.json', vocab, max_len=256)
val_dataset = FinanceDataset('val.json', vocab, max_len=256)


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4,pin_memory=True, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size,num_workers=4,pin_memory=True, shuffle=False, collate_fn=collate_fn)

# model initiating for training
encoder = Encoder(len(vocab), config.embedding_dim, config.hidden_dim).to(device)
decoder = Decoder(len(vocab), config.embedding_dim, config.hidden_dim, len(vocab)).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

print("Model is on:", next(model.parameters()).device)

# Loss = NLLLoss and Optimiser = Adam
criterion = nn.NLLLoss(ignore_index=vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.epochs):
    print(f"\nStarting epoch {epoch+1}/{config.epochs}...")
    model.train()
    total_loss = 0

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, config.teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))

    # validation rest
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))

    # Log metrics to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": avg_train_loss,
        "Train Perplexity": train_perplexity.item(),
        "Val Loss": avg_val_loss,
        "Val Perplexity": val_perplexity.item()
    })

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Train PPL = {train_perplexity:.2f}, Val PPL = {val_perplexity:.2f}")

    # model saving after every epoch as the model was training rate was too slow
    model_filename = f'seq2seq_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), model_filename)

    artifact = wandb.Artifact(f'seq2seq-finance-model-epoch-{epoch+1}', type='model')
    artifact.add_file(model_filename)
    wandb.log_artifact(artifact)

wandb.finish()
