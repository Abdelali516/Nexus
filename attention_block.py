import torch.nn as nn
from layers.rope import RoPE
import torch.nn.functional as F
import torch

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len, base):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads= num_heads
        self.num_kv_heads= num_kv_heads
        self.head_dim= d_model // num_heads

        self.w_q=nn.Linear(d_model, d_model, bias=False)
        self.w_k=nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_v=nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_o=nn.Linear(d_model, d_model, bias=False)

        self.rope=RoPE(self.head_dim, max_seq_len, base)

        self.cache_k=None
        self.cache_v=None
    
    def forward(self, x, start_pos=0, use_cache=False):
        batch_size, seq_len, _ = x.shape

        Q=self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)

        Q=Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K=K.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V=V.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)


        Q=self.rope(Q, start_pos)
        K=self.rope(K, start_pos)

        if use_cache:
            if self.cache_k is not None: # checking only one is enough since they are both together !
                K=torch.cat([self.cache_k, K], dim=2)
                V=torch.cat([self.cache_v, V], dim=2)

            self.cache_k=K
            self.cache_v=V

        K=K.repeat_interleave( self.num_heads // self.num_kv_heads, dim=1)
        V=V.repeat_interleave( self.num_heads // self.num_kv_heads, dim=1)

        if not use_cache:
            out=F.scaled_dot_product_attention(Q, K, V, is_causal=True) # in training or a full sequence with no cache (standard casual mask)

        elif seq_len==1:
            out=F.scaled_dot_product_attention(Q, K, V, is_causal=False) # one new token attends to every cached position : no mask needed 

        else:
            q_len=Q.shape[2]
            k_len=K.shape[2]

            offset= k_len - q_len # number of cahed positions

            mask=torch.ones(q_len, k_len, dtype=torch.bool, device=x.device).tril(diagonal=offset)
            out=F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

        out=out.transpose(1, 2).reshape(batch_size, seq_len, self.head_dim * self.num_heads)
        out=self.w_o(out)

        return out

    def reset_cache(self):
        self.cache_k=None
        self.cache_v=None


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.gate=nn.Linear(d_model, d_ff, bias=False)
        self.act=nn.SiLU()
        self.up=nn.Linear(d_model, d_ff, bias=False)
        self.down=nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down( self.act(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len, base, d_ff,norm_eps):
        super().__init__()

        self.attention=CausalSelfAttention(d_model, num_heads, num_kv_heads, max_seq_len, base)
        self.norm1=nn.RMSNorm(d_model, eps=norm_eps)
        self.mlp=SwiGLU(d_model, d_ff)
        self.norm2=nn.RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, start_pos=0, use_cache=False):
        x = x + self.attention(self.norm1(x), start_pos, use_cache)
        x = x + self.mlp(self.norm2(x))

        return x

    def reset_cache(self):
        self.attention.reset_cache()

