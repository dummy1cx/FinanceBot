
from sklearn.model_selection import train_test_split
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import Voc, normalizeString, batch2TrainData

MAX_LENGTH = 30

class ChatDataset(Dataset):
    def __init__(self, pairs, voc):
        self.pairs = pairs
        self.voc = voc

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def readVocs(datafile, corpus_name):
    print("Reading lines...")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = []
    for l in lines:

      parts = l.split('\t')
      if len(parts) == 2:

        pairs.append([normalizeString(parts[0]), normalizeString(parts[1])])

    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence, output_sentence = pair
        keep_input = all(word in voc.word2index for word in input_sentence.split(' '))
        keep_output = all(word in voc.word2index for word in output_sentence.split(' '))
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def collate_fn(batch, voc):
    return batch2TrainData(voc, batch)


def split_dataset(pairs, test_size=0.1, random_state=42):
    train_pairs, val_pairs = train_test_split(pairs, test_size=test_size, random_state=random_state)
    print(f"Split {len(pairs)} pairs into {len(train_pairs)} train and {len(val_pairs)} validation")
    return train_pairs, val_pairs


def get_dataloader(pairs, voc, batch_size=64, shuffle=True):
    dataset = ChatDataset(pairs, voc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: collate_fn(x, voc))