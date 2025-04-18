import os
import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import Voc, load_glove_embeddings
from dataset import loadPrepareData, trimRareWords, split_dataset, get_dataloader
from model import LSTMEncoder, LSTMAttnDecoder
from wandb_config import init_wandb
from datetime import datetime

dev_config = init_wandb()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datafile = "/content/formatted_pairs.txt"
save_dir = os.path.join("data", "save")
corpus_name = "custom"
voc, pairs = loadPrepareData("data", corpus_name, datafile, save_dir)
pairs = trimRareWords(voc, pairs, MIN_COUNT=dev_config.MIN_COUNT if "MIN_COUNT" in dev_config else 3)

train_pairs, val_pairs = split_dataset(pairs, test_size=0.1)

train_loader = get_dataloader(train_pairs, voc, batch_size=dev_config.batch_size)
val_loader = get_dataloader(val_pairs, voc, batch_size=dev_config.batch_size)

embedding = load_glove_embeddings(
    voc,
    glove_path="/content/glove.6B.300d.txt",
    embedding_dim=dev_config.embedding_dim,
    freeze=dev_config.freeze_embeddings
)

encoder = LSTMEncoder(dev_config.hidden_size, embedding, dev_config.encoder_n_layers, dev_config.dropout).to(device)
decoder = LSTMAttnDecoder(dev_config.attn_model, embedding, dev_config.hidden_size, voc.num_words,
                          dev_config.decoder_n_layers, dev_config.dropout).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=dev_config.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=dev_config.learning_rate * dev_config.decoder_learning_ratio)

def save_checkpoint_tar(voc, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, iteration, loss, save_path="checkpoint.tar"):
    checkpoint = {
        'iteration': iteration,
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'embedding_state': embedding.state_dict(),
        'encoder_optimizer_state': encoder_optimizer.state_dict(),
        'decoder_optimizer_state': decoder_optimizer.state_dict(),
        'voc_dict': voc.__dict__,
        'loss': loss
    }
    torch.save(checkpoint, save_path)
    with open("voc.pkl", "wb") as f:
        pickle.dump(voc, f)

def log_artifacts_to_wandb(tar_path="checkpoint.tar", voc_path="voc.pkl", artifact_name="chatbot_model"):
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(tar_path)
    artifact.add_file(voc_path)
    wandb.log_artifact(artifact)

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, clip):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    lengths = lengths.to("cpu")

    current_batch_size = input_variable.size(1)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_variable, lengths)
    decoder_input = torch.LongTensor([[1 for _ in range(current_batch_size)]]).to(device)
    decoder_hidden = (encoder_hidden[:decoder.n_layers], encoder_cell[:decoder.n_layers])

    use_teacher_forcing = True if torch.rand(1).item() < dev_config.teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(current_batch_size)]]).to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), dev_config.clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), dev_config.clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def evaluate_loss(val_loader, encoder, decoder, embedding):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for input_variable, lengths, target_variable, mask, max_target_len in val_loader:
            input_variable = input_variable.to(device)
            target_variable = target_variable.to(device)
            mask = mask.to(device)
            lengths = lengths.to("cpu")

            current_batch_size = input_variable.size(1)

            encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_variable, lengths)
            decoder_input = torch.LongTensor([[1 for _ in range(current_batch_size)]]).to(device)
            decoder_hidden = (encoder_hidden[:decoder.n_layers], encoder_cell[:decoder.n_layers])

            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                total_loss += mask_loss.item() * nTotal
                total_count += nTotal
    encoder.train()
    decoder.train()
    return total_loss / total_count

print("\nStarting training...")
train_iter = iter(train_loader)
for iteration in range(1, dev_config.n_iteration + 1):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    input_variable, lengths, target_variable, mask, max_target_len = batch
    train_loss = train(input_variable, lengths, target_variable, mask, max_target_len,
                       encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
                       dev_config.clip)

    perplexity = math.exp(train_loss)
    wandb.log({
        "train_loss": train_loss,
        "train_perplexity": perplexity,
        "iteration": iteration
    })

    if iteration % dev_config.print_every == 0:
        print("Iteration: {}; Train Loss: {:.4f} | Perplexity: {:.4f}".format(iteration, train_loss, perplexity))

    if iteration % dev_config.save_every == 0:
        val_loss = evaluate_loss(val_loader, encoder, decoder, embedding)
        val_perplexity = math.exp(val_loss)
        wandb.log({
            "val_loss": val_loss,
            "val_perplexity": val_perplexity
        })
        print("Validation Loss: {:.4f} | Perplexity: {:.4f}".format(val_loss, val_perplexity))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_checkpoint_tar(voc, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, iteration, val_loss, f"checkpoint_{timestamp}.tar")
        log_artifacts_to_wandb(f"checkpoint_{timestamp}.tar", "voc.pkl", f"chatbot_checkpoint_{iteration}")

save_checkpoint_tar(voc, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, iteration, train_loss, "final_checkpoint.tar")
log_artifacts_to_wandb("final_checkpoint.tar", "voc.pkl", "final_chatbot_checkpoint")
print("Final checkpoint saved to W&B.")
