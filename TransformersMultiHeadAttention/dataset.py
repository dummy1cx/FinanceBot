import pandas as pd
import torch
import  re
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from tokenizer import Vectorization
import pandas as pd



# Define device at the top
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")

class datasetLoader():

   def __init__(self,path):
       self.path = path

   def read_json(self):
       # Convert to DataFrame
       data = pd.DataFrame(pd.read_json(self.path))
       df_cleaned = (data)

       # Drop 'input' and 'text' columns
       df_cleaned = df_cleaned.drop(columns=['input', 'text'])

       return df_cleaned

   def dialog_treatment(self,df, column):
       df[column] = df[column].astype(str)  # Convert all the values of the column to str type
       df[column] = df[column].str.lower()  # Transform the string values in lowercase
       df[column] = df[column].apply(
           lambda x: re.sub("[^A-Za-z\s]", "", x))  # Replace any non-alphabetical characters with white space
       df[column] = df[column].apply(lambda x: x.replace("\s+", " "))  # Replace white spaces with a single white space
       df[column] = df[column].apply(lambda x: " ".join([word for word in x.split()]))  ###'''
       return df

   def answer_treatment(self,df, column):
       df[column] = df[column].astype(str)  # Convert all the values of the column to str type
       df[column] = df[column].str.lower()  # Transform the string values in lowercase
       df[column] = df[column].apply(lambda x: re.sub(r'\d', '', x))  # Remove all numeric characters
       df[column] = df[column].apply(lambda x: x.replace("\s+", " "))  # Replace white spaces with a single white space
       df[column] = df[column].apply(lambda x: re.sub(r"[-()&\"#/@;:<>{}`+=~|.!?,।]", "",
                                                      x))  # Remove specific punctuation and special characters from the values
       df[column] = df[column].apply(lambda x: x.strip())  # Clean up text data  removing unwanted spaces'''
       df[column] = "<sos> " + df[column] + " <eos>"  # Add start of sequence and end of sequence to the string
       return df

   def treatment(self):
       df = self.read_json()
       df['output'] = df['output'].apply(lambda x: ' '.join(x.split()[:20]))

       df['src_len'] = [len(text.split()) for text in df.instruction]
       df['trg_len'] = [len(text.split()) for text in df.output]

       df = self.dialog_treatment(df, 'instruction')
       df = self.answer_treatment(df, 'output')

       return df


   def vectorization(self):
       df = self.treatment()
       src_max_lenght_sentence = np.max(df['src_len'])
       trg_max_lenght_sentence = np.max(df['trg_len'])

       src_sequences, src_tokenizer = Vectorization(df,'instruction', src_max_lenght_sentence)
       trg_sequences, trg_tokenizer = Vectorization(df,'output', trg_max_lenght_sentence)

       print("Size of the source vocabulary :", len(src_tokenizer.word_index))
       print("Size of the target vocabulary :", len(trg_tokenizer.word_index))

       # Verify sequence tokenized
       trg_sent = ' '.join([trg_tokenizer.index_word[idx] for idx in trg_sequences[6] if idx != 0])
       print(f"{trg_sequences[6]} \n\n {trg_sent}")

       return src_sequences, trg_sequences, src_tokenizer, trg_tokenizer

   def  define_dataloader(self):
       df = self.treatment()

       batch_size = 128

       src_sequences, trg_sequences, src_tokenizer, trg_tokenizer = self.vectorization()

       dataset = TensorDataset(torch.LongTensor(src_sequences), torch.LongTensor(
           trg_sequences))

       torch.manual_seed(42)

       dataframe_dataloader = DataLoader(
           dataset=dataset,
           batch_size=batch_size,
           shuffle=True,
           num_workers=4,
           pin_memory=True
       )
       src_vocab_size = len(src_tokenizer.word_index)  # Vocabulary size for source
       trg_vocab_size = len(trg_tokenizer.word_index)  # Vocabulary size for target
       src_max_len = np.max(df['src_len'])
       trg_max_len = np.max(df['trg_len'])

       return dataframe_dataloader, src_vocab_size, trg_vocab_size,src_max_len,trg_max_len,src_tokenizer,trg_tokenizer










