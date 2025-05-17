import numpy as np
import torch
from TransformersMultiHeadAttention.dataset import datasetLoader

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")

def evaluate(sent):

    sentence = src_tokenizer.texts_to_sequences([sent])[0]  # Get the list of token ids

    # Pad sentence manually with PyTorch
    if len(sentence) < SRC_MAXLEN:
        sentence = sentence + [0] * (SRC_MAXLEN - len(sentence))  # Pad with 0s (post-padding)
    else:
        sentence = sentence[:SRC_MAXLEN]

    # Convert to tensor
    src_input = torch.tensor([sentence], dtype=torch.int64).to(DEVICE)

    # Decoder input starts with SOS token
    decoder_input = [trg_tokenizer.word_index['sos']]
    decoder_input = torch.tensor([decoder_input], dtype=torch.int64).to(DEVICE)

    # Generate prediction
    for i in range(TRG_MAXLEN):
        preds = model(src_input, decoder_input)  # (batch, seq_len, vocab_size)

        # Get the last token prediction
        preds = preds[:, -1:, :]  # Get prediction at last time step
        predicted_id = torch.argmax(preds, dim=-1)  # (batch, 1)

        # Stop if EOS
        if predicted_id.item() == trg_tokenizer.word_index['eos']:
            return decoder_input.squeeze(0)

        # Concatenate predicted token to decoder input
        decoder_input = torch.cat([decoder_input, predicted_id], dim=1)

    return decoder_input.squeeze(0)


# Load data and tokenizers
path = "./Cleaned_date.json"
loader = datasetLoader(path)
df = loader.treatment()
src_sequences, trg_sequences, src_tokenizer, trg_tokenizer = loader.vectorization()

# Set max sequence lengths
SRC_MAXLEN = int(np.max(df['src_len']))
TRG_MAXLEN = int(np.max(df['trg_len']))

# Sample test data
test_sample = df.head(100)
x_test = test_sample['instruction'].tolist()
y_test = test_sample['output'].tolist()

# Import model and load it
from TransformersMultiHeadAttention.transformer import Transformer

num_layers = 4
model_dimension = 256
inner_layer_dimension = 512
num_heads = 8
dropout_rate = 0.1

model = Transformer(
    num_layers,
    model_dimension,
    num_heads,
    inner_layer_dimension,
    len(src_tokenizer.word_index) + 1,  # src_vocab_size
    len(trg_tokenizer.word_index) + 1,  # trg_vocab_size
    SRC_MAXLEN,
    TRG_MAXLEN,
    dropout_rate
).to(DEVICE)

# Load trained weights
model.load_state_dict(torch.load("./transformer_model_.pt", map_location=DEVICE))
model.eval()


# Chatbot interface
def chatbot():
    print("Welcome to FinanceBot!")
    print("Type 'exit' to quit.\n")

    while True:
        src_sent = input("Type: ")
        if src_sent.lower() == 'exit':
            print("Goodbye!")
            break

        # Run the model evaluation
        result = evaluate(src_sent)

        # Convert predicted indices to words
        pred_sent = ' '.join([
            trg_tokenizer.index_word[idx]
            for idx in result.cpu().numpy()
            if idx != 0 and idx != 2  # Filter out padding and special tokens
        ])

        print(f"Output: {pred_sent}\n")


# Run chatbot
chatbot()
