from torch import nn
from MQA import MultiQueryAttention
import torch

# Define device at the top
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")



class DecoderLayer(nn.Module) :
    def __init__(self , model_dimension , num_heads , inner_layer_dimension , dropout_rate = 0.1) :
        super(DecoderLayer , self).__init__()

        self.self_attn = MultiQueryAttention(model_dimension , num_heads)

        # Multi-head attention mechanism that attends to the encoder's output

        self.cross_attn = MultiQueryAttention(model_dimension , num_heads)

        # Layer normalization
        self.norm1 = nn.LayerNorm(model_dimension , eps=1e-6)
        self.norm2 = nn.LayerNorm(model_dimension , eps=1e-6)
        self.norm3 = nn.LayerNorm(model_dimension , eps=1e-6)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.feed_forward_network = nn.Sequential( ######
            nn.Linear(model_dimension, inner_layer_dimension),
            nn.ReLU() ,
            nn.Linear(inner_layer_dimension, model_dimension))

    def forward(self , x , enc_output , src_mask , padding_mask) :

        attention_output = self.self_attn(x , x , x , src_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.norm1(attention_output  + x)

        attn2 = self.cross_attn(out1 , enc_output , enc_output , padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.norm2(attn2 + out1)

        ff_output = self.dropout3(self.feed_forward_network(out2))
        decoder_output = self.norm3(ff_output + out2)

        return decoder_output

class Decoder(nn.Module):
        def __init__(self, num_layers, model_dimension, num_heads, inner_layer_dimension, trg_vocab_size, max_length,
                     dropout_rate=0.1):
            super(Decoder, self).__init__()

            self.num_layers = num_layers

            self.embedding = nn.Embedding(trg_vocab_size, model_dimension)

            self.pos_encoding = nn.Embedding(max_length, model_dimension)

            self.decoder_layers = [
                DecoderLayer(model_dimension, num_heads, inner_layer_dimension, dropout_rate).to(DEVICE) for _ in
                range(num_layers)]

            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, encoder_output, src_mask, padding_mask):  ########
            batch_size, seqlen = x.shape

            positions = torch.arange(0, seqlen).expand(batch_size, seqlen).to(DEVICE)

            output = self.dropout((self.embedding(x) + self.pos_encoding(positions)))

            # Applies each decoder layer sequentially to the output of the previous layer
            for i in range(self.num_layers): output = self.decoder_layers[i](output, encoder_output, src_mask,
                                                                             padding_mask)

            return output