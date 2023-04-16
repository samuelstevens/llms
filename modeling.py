"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import dataclasses
import os

import sentencepiece
import torch
import torch.nn as nn
from torch.nn import functional as F


def build_rope_cache(n_elem: int, seq_len: int, theta: int = 10000) -> torch.Tensor:
    theta = 1.0 / (theta ** (torch.arange(0, n_elem, 2).float() / n_elem))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, theta).float()
    cache = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return cache


def apply_rope(x, rope_cache):
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because `view_as_complex` does not support 16 bit tensors
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config, rope_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("rope_cache", rope_cache, persistent=False)

    def forward(self, x, *, attn_mask=None):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch.
        # then move head forward to be the batch dim.
        q, k, v = self.wqkv(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        # (sam) Why view then transpose?
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        if attn_mask is not None:
            # Insert a new batch dimension for the multiple attn heads
            attn_mask = attn_mask.view(B, 1, T, T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # We use 2/3 4d instead of 4d as in PaLM
        # Round to the nearest multiple of 256
        hidden_dim = int(2 / 3 * 4 * config.n_embd)
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config, rope_cache):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(config, rope_cache)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x, *, attn_mask=None):
        x = x + self.attention(self.attention_norm(x), attn_mask=attn_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


@dataclasses.dataclass
class Config:
    name: str = "llama-7b"
    # model
    max_seq_len: int = 8192  # can be quite large; only affects rope cache
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    # tokenizer
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 2  # by default use eos_id for pad_id
    # TODO: pad this to a nearest multiple of 256 for efficiency
    vocab_size: int = 32000

    @classmethod
    def from_name(cls, name: str):
        return llama_configs[name]


llama_configs = {
    "llama-7b": Config(n_layer=32, n_head=32, n_embd=4096),
    "llama-13b": Config(name="llama-13b", n_layer=40, n_head=40, n_embd=5120),
    "llama-30b": Config(name="llama-30b", n_layer=60, n_head=52, n_embd=6656),
    "llama-65b": Config(name="llama-65b", n_layer=80, n_head=64, n_embd=8192),
}


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        rope_cache = build_rope_cache(
            n_elem=config.n_embd // config.n_head,
            seq_len=config.max_seq_len,
        )

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            [Block(config, rope_cache) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Includes token embedding parameters, which we normally don't.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, toks, *, targets=None, loss_mask=None, attn_mask=None):
        b, t = toks.shape

        # token embeddings of shape (b, t, n_embd)
        x = self.tok_embeddings(toks)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(x)
            
            if loss_mask is not None:
                logits = logits[loss_mask == 1]
                targets = targets[loss_mask == 1]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.output(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, toks, max_new_tokens, *, temp=1.0, top_k=None):
        """
        Take a conditioning sequence of tokens (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Most likely you'll want to make sure to
        be in model.eval() mode of operation for this.
        """
        attn_mask = None
        for _ in range(max_new_tokens):
            # if the context is growing too long we must crop it at max_seq_len
            tok_ctx = toks[:, -self.config.max_seq_len :]
            # TODO: switch 2 to the padding token
            attn_mask = pad_for_gen(tok_ctx, self.config.pad_id, prev_mask=attn_mask)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(tok_ctx, attn_mask=attn_mask)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temp
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_tok = torch.multinomial(probs, num_samples=1)
            # append sampled tokens to the running sequence and continue
            toks = torch.cat((toks, next_tok), dim=1)

        return toks


def pad_for_gen(tokens, pad_id, *, prev_mask=None):
    inf = 1e9  # Use 1e9 as infinity to avoid nan

    B, T = tokens.shape

    if prev_mask is not None:
        mask = prev_mask
        # Add column on the right
        mask = F.pad(mask, (0, 1, 0, 0), "constant", -inf)
        # Add row on the bottom
        mask = F.pad(mask, (0, 0, 0, 1), "constant", 0.0)
    else:
        # 1. 0s in the upper diagonal, 1s in the lower
        # 2. Set all 0s to -infinity
        # 3. Set all 1s to 0
        # 4. End with -infinity in the upper diagonal, 0s in the lower
        mask = torch.ones((B, T, T), device=tokens.device).tril()
        mask = mask.masked_fill(mask == 0, -inf)
        mask = mask.masked_fill(mask == 1, 0)

    assert mask.shape == (B, T, T), "mask is wrong shape!"

    pad_tokens = (tokens == pad_id).unsqueeze(1)  # (B, 1, T)
    return mask.masked_fill(pad_tokens, -inf)


class Tokenizer:
    def __init__(self, config, sp_model):
        self.config = config
        self.sp_model = sp_model
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, msg: str, *, bos: bool, eos: bool, out: str = "py") -> list[int]:
        assert type(msg) is str
        t = self.sp_model.encode(msg)
        if bos:
            t = [self.config.bos_id] + t
        if eos:
            t = t + [self.config.eos_id]
        if out == "py":
            return t
        elif out == "pt":
            return torch.tensor(t, dtype=torch.int64)
        else:
            raise ValueError(out)

    def encode_batch(self, msgs: list[str], *, bos: bool, eos: bool) -> torch.Tensor:
        if eos:
            raise NotImplementedError("batch encoding with EOS token")

        batch = [self.encode(s, bos=bos, eos=eos) for s in msgs]
        B = len(batch)
        max_len = max(len(t) for t in batch)

        padded = torch.full((B, max_len), self.config.pad_id)
        for b, tokens in enumerate(batch):
            padded[b, -len(tokens) :] = torch.tensor(tokens)

        return padded

    def decode(self, t: list[int]) -> str:
        return self.sp_model.decode(t)

    def decode_batch(self, toks: torch.Tensor):
        # Need to ignore any tokens after eos_id
        raise NotImplementedError()


@torch.no_grad()
def load_pretrained_llama(config, model_ckpt):
    print(f"(1/4) Loading pretrained weights {config.name}.")

    # n_layer, n_head and n_embd are determined from name
    # create a from-scratch initialized transformer
    model = Llama(config)
    print(f"(2/4) Initialized {config.name} on CPU.")

    checkpoint = torch.load(model_ckpt, map_location="cpu")
    print(f"(3/4) Loaded pretrained {config.name} onto CPU.")

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)

    # Put the wq, wk and wv matrices into a single matrix called wqkv
    # Ignore those keys in unexpected (used)
    used = set()
    found = set()
    for i, layer in enumerate(model.layers):
        key = f"layers.{i}.attention.wq.weight"
        layer.attention.wqkv.weight[: config.n_embd, :] = checkpoint[key]
        used.add(key)

        key = f"layers.{i}.attention.wk.weight"
        layer.attention.wqkv.weight[config.n_embd : config.n_embd * 2, :] = checkpoint[
            key
        ]
        used.add(key)

        key = f"layers.{i}.attention.wv.weight"
        layer.attention.wqkv.weight[
            config.n_embd * 2 : config.n_embd * 3, :
        ] = checkpoint[key]
        used.add(key)

        found.add(f"layers.{i}.attention.wqkv.weight")

    print("(4/4) Loaded state dict.")

    missing = [k for k in missing if k not in found]
    if missing:
        print(f"Missing keys: {missing}")
    unexpected = [k for k in missing if k not in used]
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model


def load_pretrained_tokenizer(config, tok_ckpt):
    # reload tokenizer
    assert os.path.isfile(tok_ckpt), tok_ckpt
    sp_model = sentencepiece.SentencePieceProcessor(model_file=tok_ckpt)
    assert config.vocab_size == sp_model.vocab_size()

    # BOS / EOS token IDs
    assert config.bos_id == sp_model.bos_id()
    assert config.eos_id == sp_model.eos_id()
    # Don't check pad token because it doesn't matter

    tokenizer = Tokenizer(config, sp_model)

    return tokenizer
