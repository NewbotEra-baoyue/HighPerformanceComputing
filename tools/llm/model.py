"""手搓 Decoder-only Transformer(GPT 风格),不依赖 nn.Transformer"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # (B, T, H, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        att = att.masked_fill(causal_mask, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        out = att @ v  # (B, H, T, D)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x, causal_mask):
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)
        )
        self.ln_f = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 权重 tying
        self.apply(self._init)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), 1),
            persistent=False,
        )

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb.weight[:T]
        x = self.drop(x)
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.pos_emb.num_embeddings :])[:, -1]
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = logits.topk(top_k, dim=-1)
                    logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:  # greedy
                nxt = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx


if __name__ == "__main__":
    m = GPT(vocab_size=60)
    x = torch.randint(0, 60, (2, 16))
    print(m(x).shape)
    print(m.generate(x[:, :1], 8).shape)
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
