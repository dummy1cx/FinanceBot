import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Tokenizer
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data):
    for item in data:
        yield tokenizer(item['instruction'] + ' ' + item['output'])


# build Vocabulary from the dataset
# minimal to no text pre processing done 
# the idea is to feed un processed word to the model to understand model behaviour
def build_vocab(data):
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=['<pad>', '<sos>', '<eos>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


# encoding sentences
def encode_sentence(sentence, vocab, max_len=128):
    tokens = tokenizer(sentence)[:max_len]
    token_ids = [vocab['<sos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
    return torch.tensor(token_ids)



def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src, trg in batch:
        src_batch.append(src)
        trg_batch.append(trg)

    src_batch = pad_sequence(src_batch, padding_value=0)
    trg_batch = pad_sequence(trg_batch, padding_value=0)

    return src_batch, trg_batch

