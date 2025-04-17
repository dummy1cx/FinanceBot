import torch
from torch.utils.data import Dataset
from utils import encode_sentence, load_json

class FinanceDataset(Dataset):
    def __init__(self, json_file, vocab, max_len=256):
        self.data = load_json(json_file)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_sentence = item['instruction']
        trg_sentence = item['output']

        src_tensor = encode_sentence(src_sentence, self.vocab, max_len=self.max_len)
        trg_tensor = encode_sentence(trg_sentence, self.vocab, max_len=self.max_len)

        return src_tensor, trg_tensor
