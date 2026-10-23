# Nexus Layers

Building blocks for the Nexus model: the token embedding that turns token ids into vectors, and
RoPE, which encodes token positions inside attention.

## Pipeline

<p align="center">
  <img src="../assets/layers_pipeline.png" alt="Nexus Layers Pipeline" width="550">
</p>

## Contents

- `embedding.py`: `TokenEmbedding`, token ids to vectors
- `rope.py`: `RoPE`, position rotation for queries and keys


## Token embedding (`embedding.py`)

`TokenEmbedding(vocab_size, d_model, init_std)` maps token ids of shape `(B, T)` to vectors of
shape `(B, T, d_model)` by looking up rows in a weight table of shape `(vocab_size, d_model)`.

**Initialization:** the weight is drawn from a normal distribution with `mean=0.0` and
`std=init_std` (0.02 from `config.json`).

- **Mean 0** means the token vectors start centered on zero, with no shared offset that would
  carry no information about which token is which.
- **Std 0.02** keeps the starting values small, so the first layers receive modest inputs.
  PyTorch's default `nn.Embedding` uses std 1, which is about 50 times larger.

The weight stays in float32. bfloat16 is applied by autocast during training, so the optimizer
updates keep full precision. The weight will be shared with the output head (`tie_embeddings`).

There is no positional table here. Position is applied inside attention by RoPE.

## RoPE (`rope.py`)

`RoPE(head_dim, max_seq_len, base)` rotates queries and keys by their position. Values are not
rotated.

- **Frequencies:** `inv_freq[i] = 1 / base^(2i / head_dim)` for `i` from 0 to `head_dim/2 - 1`.
  Small `i` rotates fast and tracks nearby tokens. Large `i` rotates slowly and tracks distant
  ones.
- **Angles:** `position × inv_freq`. The `cos` and `sin` tables are computed once, for every
  position up to `max_seq_len`, and stored as buffers with `persistent=False`. They move with the
  model to the GPU and are not saved in checkpoints.
- **Pairing:** each head's first half is paired with its second half (the rotate-half
  convention). This must stay the same for training and inference.
- **`start_pos`:** the offset of the first position. During generation with a KV cache, a new
  token at position `t` is rotated with `start_pos=t`.
- **Precision:** the rotation runs in float32, and the output is cast back to the input dtype.

Defaults come from `config.json`: `head_dim = d_model / num_heads = 128`,
`max_seq_len = 4096`, and `base = rope_theta = 10000`.