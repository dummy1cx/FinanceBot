import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import indexesFromSentence, normalizeString
from dataset import MAX_LENGTH

# Helper to move tensor to correct device
def to_device(tensor):
    return tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Safe version that logs unknown words
def safe_indexesFromSentence(voc, sentence):
    missing = []
    indexes = []
    for word in sentence.split(" "):
        if word in voc.word2index:
            indexes.append(voc.word2index[word])
        else:
            missing.append(word)
    if missing:
        print(f"❌ Missing words in vocab: {missing}")
    return indexes + [2]  # EOS_token

# Greedy decoder
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # For LSTM: encoder_hidden is a tuple (h, c)
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        decoder_input = torch.ones(1, 1, device=input_seq.device, dtype=torch.long) * 1  # SOS_token

        all_tokens = torch.zeros([0], device=input_seq.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=input_seq.device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores

# Final evaluation wrapper
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    sentence = normalizeString(sentence)
    indexes_batch = [safe_indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = to_device(input_batch)
    lengths = lengths.to("cpu")

    with torch.no_grad():
        tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words