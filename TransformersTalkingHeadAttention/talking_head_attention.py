import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TakingHeadAttention(nn.Module):
    def __init__(self, model_dimension, num_heads):
        super().__init__()

        self.model_dimension = model_dimension
        self.num_heads = num_heads
        assert model_dimension % num_heads == 0

        self.head_dimension = model_dimension // num_heads

        self.q_proj = nn.Linear(model_dimension, model_dimension)
        self.k_proj = nn.Linear(model_dimension, model_dimension)
        self.v_proj = nn.Linear(model_dimension, model_dimension)
        self.out_proj = nn.Linear(model_dimension, model_dimension)

        # Talking-Heads: pre-softmax head mixing (via learned projection)
        self.logit_proj_pre = nn.Parameter(torch.randn(num_heads, num_heads))

    def split_heads(self, x, batch_size):
        return x.reshape(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.size()

        q = self.split_heads(self.q_proj(q), B)
        k = self.split_heads(self.k_proj(k), B)
        v = self.split_heads(self.v_proj(v), B)

        # Scaled dot-product attention logits
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dimension)  # [B, H, L, L]

        # Apply pre-softmax mixing: logits → [B, L, L, H] → [B, L, L, H] (mixed heads)
        logits = logits.permute(0, 2, 3, 1)  # → [B, L, L, H]
        logits = torch.matmul(logits, self.logit_proj_pre)  # Linear mix over heads
        logits = logits.permute(0, 3, 1, 2)  # → [B, H, L, L]

        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(logits, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        out = attn_output.transpose(1, 2).contiguous().view(B, L, self.model_dimension)
        return self.out_proj(out)