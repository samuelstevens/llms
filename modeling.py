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

    def forward(self, x):
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

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        # By specifying is_causal, we don't have to pass a mask.
        # But we might want a causal mask if we have padding tokens in the inputs.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
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

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


@dataclasses.dataclass
class Config:
    max_seq_len: int = 4096
    # TODO: pad this to a nearest multiple of 256 for efficiency
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    @classmethod
    def from_name(cls, name: str):
        return llama_configs[name]


llama_configs = {
    "llama-7b": Config(n_layer=32, n_head=32, n_embd=4096),
    "llama-13b": Config(n_layer=40, n_head=40, n_embd=5120),
    "llama-30b": Config(n_layer=60, n_head=52, n_embd=6656),
    "llama-65b": Config(n_layer=80, n_head=64, n_embd=8192),
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

    def forward(self, idx, targets=None):
        b, t = idx.size()

        # token embeddings of shape (b, t, n_embd)
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.output(x[:, [-1], :])
            loss = None

        return logits, loss

    @classmethod
    def from_name(cls, name):
        return cls(Config.from_name(name))

    @classmethod
    @torch.no_grad()
    def from_pretrained(cls, name, model_ckpt):
        print(f"(1/3) Loading pretrained weights {name}.")

        # n_layer, n_head and n_embd are determined from name
        # create a from-scratch initialized transformer
        config = Config.from_name(name)
        model = cls(config)
        print(f"(2/3) Initialized {name} on CPU.")

        checkpoint = torch.load(model_ckpt, map_location="cpu")
        print(f"(3/3) Loaded pretrained {name} onto CPU.")

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
            layer.attention.wqkv.weight[
                config.n_embd : config.n_embd * 2, :
            ] = checkpoint[key]
            used.add(key)

            key = f"layers.{i}.attention.wv.weight"
            layer.attention.wqkv.weight[
                config.n_embd * 2 : config.n_embd * 3, :
            ] = checkpoint[key]
            used.add(key)

            found.add(f"layers.{i}.attention.wqkv.weight")

        missing = [k for k in missing if k not in found]
        if missing:
            print(f"Missing keys: {missing}")
        unexpected = [k for k in missing if k not in used]
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and rmsnorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        return optimizer

    # ---------------------------------------------------------
    # This stuff hasn't been updated with rotary embeddings yet
    # ---------------------------------------------------------

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Most likely you'll want to make sure to
        be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_seq_len
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_seq_len
                else idx[:, -self.config.max_seq_len :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = sentencepiece.SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, *, bos: bool, eos: bool) -> list[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_batch(self, msgs: list[str], *, bos: bool, eos: bool) -> list[list[int]]:
        if eos:
            raise NotImplementedError(
                "Haven't implemented batch encoding with an EOS token"
            )

        batch = [self.encode(s, bos=bos, eos=eos) for s in msgs]

        max_len = max(len(t) for t in batch)

        return [self._pad(t, max_len) for t in batch]

    def _pad(self, t: list[int], length: int) -> list[int]:
        missing = length - len(t)
        if missing < 0:
            raise RuntimeError("Can't apply negative padding")

        return t + [self.pad_id] * missing

    def decode(self, t: list[int]) -> str:
        return self.sp_model.decode(t)
