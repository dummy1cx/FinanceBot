from torch import nn
import torch
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, model_dimension, num_heads):
        super(MultiQueryAttention, self).__init__()

        self.model_dimension = model_dimension
        self.num_heads = num_heads

        assert self.model_dimension % self.num_heads == 0
        self.head_dimension = self.model_dimension // self.num_heads

        # Each head has its own query projection
        self.Q = nn.Linear(model_dimension, model_dimension)

        # Keys and values are shared across all heads
        self.K = nn.Linear(model_dimension, self.head_dimension)
        self.V = nn.Linear(model_dimension, self.head_dimension)

        # Final output linear layer
        self.output_t = nn.Linear(model_dimension, model_dimension)

    def split_heads(self, x, batch_size):
        # (B, seq_len, model_dim) -> (B, num_heads, seq_len, head_dim)
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dimension)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]


        q_transform = self.split_heads(self.Q(q), batch_size)

        # Key and Value are shared across heads
        k_shared = self.K(k).unsqueeze(1)  # (B, 1, seq_len, head_dim)
        v_shared = self.V(v).unsqueeze(1)  # (B, 1, seq_len, head_dim)

        # Broadcast to all heads
        k_transform = k_shared.expand(-1, self.num_heads, -1, -1)
        v_transform = v_shared.expand(-1, self.num_heads, -1, -1)

        # Scaled dot-product attention
        dk = self.head_dimension
        scores = torch.matmul(q_transform, k_transform.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        scores = torch.clamp(scores, -1e6, 1e6)  # Clip to prevent large values

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        '''if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))'''

        attn_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attn_weights, v_transform)

        # Recombine heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention_output.view(batch_size, -1, self.model_dimension)

        output = self.output_t(concat_attention)
        return output