import torch.nn as nn
from layers.embedding import TokenEmbedding
from attention_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, init_std, num_heads, num_kv_heads,
                 max_seq_len, base, norm_eps, d_ff, num_layers):
        super().__init__()

        self.embedding = TokenEmbedding(vocab_size, d_model, init_std)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_kv_heads,
                             max_seq_len, base, d_ff, norm_eps)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(d_model, eps=norm_eps)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.embedding.weight

        self._init_weights(init_std, num_layers)

    def forward(self, input_ids, start_pos=0, use_cache=False):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, start_pos, use_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()

    def _init_weights(self, init_std, num_layers):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

        residual_std = init_std / (2 * num_layers) ** 0.5
        for layer in self.layers:
            nn.init.normal_(layer.attention.w_o.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down.weight, mean=0.0, std=residual_std)