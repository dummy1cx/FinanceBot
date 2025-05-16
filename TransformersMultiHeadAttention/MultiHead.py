from torch import nn
from models import  SelfAttention

class MultiHeadAttention(nn.Module) :
    def __init__(self , model_dimension , num_heads) :
        super(MultiHeadAttention, self).__init__()

        self.model_dimension = model_dimension
        self.num_heads = num_heads

        # Verify if the model dimension is divisible by num_head
        assert self.model_dimension % self.num_heads == 0

        self.head_dimension = self.model_dimension // self.num_heads # Dimension of each head's key, query, and value

        # Linear layers for input transformations
        self.Q = nn.Linear(model_dimension , model_dimension)
        self.K = nn.Linear(model_dimension , model_dimension)
        self.V = nn.Linear(model_dimension , model_dimension)
        self.output_t = nn.Linear(model_dimension , model_dimension)

    def split_heads(self , x , batch_size) :

        x = x.reshape(batch_size , -1 , self.num_heads , self.head_dimension)
        return x.permute(0, 2, 1, 3)

    def forward(self , q , k , v , mask = None) :
        batch_size = q.shape[0]

        # Apply linear transformations and split heads
        q_transform = self.split_heads(self.Q(q),batch_size)
        k_transform = self.split_heads(self.K(k),batch_size)
        v_transform = self.split_heads(self.V(v),batch_size)

        # Perform scaled dot-product attention
        self_attention = SelfAttention(q_transform , k_transform , v_transform , mask)

        # Rearrange shape
        self_attention =  self_attention.permute(0 , 2 , 1 , 3)

        # Concatenation of heads
        attention_output = self_attention.reshape(batch_size, -1, self.model_dimension)

        output = self.output_t(attention_output)

        return output