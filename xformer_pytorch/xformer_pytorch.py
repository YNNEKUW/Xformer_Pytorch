import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from quant_noise import quant_noise


class Xformer(nn.Module):
    def __init__(
        self, 
        input_dim, 
        num_heads, 
        dropout=0.0, 
        scalar=2, 
        max_seq_len=280, 
        bias=True, 
        q_noise=0.0, 
        qn_block_size=8):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scalar = scalar
        self.head_dim = (input_dim // num_heads) // scalar
        assert (
            input_dim == self.head_dim * num_heads * scalar
        ), "input_dim must be divisible by num_heads and scalar."
        self.scaling = (input_dim // num_heads) ** -0.5

        self.k_v_proj = quant_noise(nn.Linear(max_seq_len, max_seq_len//scalar, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(input_dim, input_dim//scalar, bias=bias), q_noise, qn_block_size)

        self.out_proj = quant_noise(nn.Linear(input_dim, input_dim, bias=bias), q_noise, qn_block_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, input_tensor):
        input_length, batch_size, hidden_dim = input_tensor.shape
        # X = input_tensor.flatten(2)

        X = rearrange(input_tensor, 'l b d -> b l d')
        q = self.q_proj(X)
        v = rearrange(X, 'b l d -> b d l') @ self.k_v_proj.weight.T[:input_length, :input_length//self.scalar+1]
        k = rearrange(q, 'b l d -> b d l') @ self.k_v_proj.weight.T[:input_length, :input_length//self.scalar+1]
        
        q = q * self.scaling
        
        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b (d h) l -> (b h) d l', h=self.num_heads)
        v = rearrange(v, 'b (d h) l -> (b h) d l', h=self.num_heads)
        v = rearrange(v, 'b d l -> b l d')


        attn_weights = q @ k
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training
        )
        X = attn_probs @ v
        X = rearrange(X, '(b h) l d -> b l (d h)', h=self.num_heads)
        X = self.out_proj(X)
        X = rearrange(X, 'b l d -> l b d')

        return X
