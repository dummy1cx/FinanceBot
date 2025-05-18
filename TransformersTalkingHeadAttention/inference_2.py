import streamlit as st
import numpy as np
import torch
from TransformersMultiHeadAttention.transformer import Transformer
from TransformersMultiHeadAttention.dataset import datasetLoader


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "../TransformersMQA/trained_model_talking_head.pt"
DATA_PATH = "./Cleaned_date.json"


@st.cache_resource
def load_everything():
    loader = datasetLoader(DATA_PATH)
    df = loader.treatment()
    src_sequences, trg_sequences, src_tokenizer, trg_tokenizer = loader.vectorization()

    SRC_MAXLEN = int(np.max(df['src_len']))
    TRG_MAXLEN = int(np.max(df['trg_len']))

    # Model params from training
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
        len(src_tokenizer.word_index) + 1,
        len(trg_tokenizer.word_index) + 1,
        SRC_MAXLEN,
        TRG_MAXLEN,
        dropout_rate
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, src_tokenizer, trg_tokenizer, SRC_MAXLEN, TRG_MAXLEN

model, src_tokenizer, trg_tokenizer, SRC_MAXLEN, TRG_MAXLEN = load_everything()


def evaluate(sent):
    sentence = src_tokenizer.texts_to_sequences([sent])[0]

    if len(sentence) < SRC_MAXLEN:
        sentence = sentence + [0] * (SRC_MAXLEN - len(sentence))
    else:
        sentence = sentence[:SRC_MAXLEN]

    src_input = torch.tensor([sentence], dtype=torch.int64).to(DEVICE)
    decoder_input = [trg_tokenizer.word_index['sos']]
    decoder_input = torch.tensor([decoder_input], dtype=torch.int64).to(DEVICE)

    for _ in range(TRG_MAXLEN):
        preds = model(src_input, decoder_input)
        preds = preds[:, -1:, :]
        predicted_id = torch.argmax(preds, dim=-1)

        if predicted_id.item() == trg_tokenizer.word_index['eos']:
            return decoder_input.squeeze(0)

        decoder_input = torch.cat([decoder_input, predicted_id], dim=1)

    return decoder_input.squeeze(0)


st.title("💬 FinanceBot")
st.markdown("Ask a financial question or give an instruction. The model will respond.")

user_input = st.text_area("Type your instruction/question:")

if st.button("Generate Response"):
    if user_input.strip():
        result = evaluate(user_input)

        pred_sent = ' '.join([
            trg_tokenizer.index_word[idx]
            for idx in result.cpu().numpy()
            if idx != 0 and idx != 2  # Exclude pad and eos
        ])

        st.success(f"**Output:** {pred_sent}")
    else:
        st.warning("Please enter something to translate.")

st.markdown("---")
st.markdown("Type `exit` in the original CLI version to quit.")
