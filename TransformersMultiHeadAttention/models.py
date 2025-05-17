import torch

# Define device at the top
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")

class SelfAttention():
    def Self_Attention(q, k, v, mask):
        attention_scores = torch.matmul(q, k.transpose(-2, -1)).to(DEVICE)  # Matrix multiplication

        scaling = torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32)).to(
            DEVICE)  # Computes the square root of dim

        scaled_attention = attention_scores / scaling

        # Apply mask
        if mask is not None:
            scaled_attention += (mask * -1e9)

        # Apply softmax
        attention_weights = torch.softmax(scaled_attention, dim=-1).to(DEVICE)

        # Compute the weighted sum of the value vectors using the attention weights
        output = torch.matmul(attention_weights, v).to(DEVICE)

        return output
