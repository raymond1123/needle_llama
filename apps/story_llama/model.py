from dataclasses import dataclass
from typing import Optional
import math
import json
import numpy as np

import torch

from load_params import load_meta_model
import needle as ndl
import needle.nn as nn

import subprocess

def print_gpu_memory():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode())

#print_gpu_memory()

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def repeat_kv(x: ndl.Tensor, n_rep: int) -> ndl.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False) # 512*512
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False) # 512*512
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False) # 512*512
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False) # 512*512

        cache_shape = [args.max_batch_size, 
                       args.max_seq_len, 
                       self.n_kv_heads, self.head_dim]

        self.cache_k = ndl.zeros(cache_shape, ndl.fp16, ndl.cuda)
        self.cache_v = ndl.zeros(cache_shape, ndl.fp16, ndl.cuda)

    def forward (
        self,
        x: ndl.Tensor,
        start_pos: int,
    ):
        #freqs_complex: needle.Tensor
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.reshape([batch_size, seq_len, self.n_heads_q, self.head_dim])
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        #xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xq = xq.rotary_emb(start_pos)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        #xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        xk = xk.rotary_emb(start_pos)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk # (2, 1024, 8, 64)
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv # (2, 1024, 8, 64)

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose([1, 2])
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose([1, 2])
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose([1, 2])

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = xq@(keys.transpose([2, 3])) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = scores.softmax()

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = scores@values
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = output.transpose([1, 2])
        output.contiguous()
        output = output.reshape([batch_size, seq_len, -1])
        #output = output.reshape([batch_size, seq_len, 8*64])
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        if args.hidden_dim is None:
            hidden_dim = 4 * args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if args.ffn_dim_multiplier is not None:
                hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            # Round the hidden_dim to the nearest multiple of the multiple_of parameter
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        else:
            hidden_dim=args.hidden_dim

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: ndl.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        # silu: x*sigmoid(x)
        swish = self.w1(x).silu()
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.x_rms_weight = ndl.ones([args.dim])
        self.h_rms_weight = ndl.ones([args.dim])

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention block
        #self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        #self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: ndl.Tensor, start_pos: int):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(x.rms_norm(self.x_rms_weight), start_pos)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        ffn_norm = h.rms_norm(self.h_rms_weight)
        out = h + self.feed_forward.forward(ffn_norm)
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.rms_weight = ndl.ones([args.dim])

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim, 
                                           dtype=ndl.fp16, device=ndl.cuda)

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, tokens: ndl.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            layer.half()
            h = layer.forward(h, start_pos)
            #print_gpu_memory()

        h = h.rms_norm(self.rms_weight)
        output = self.output(h)
        return output

if __name__ == '__main__':

    llama_model = './story_model'

    prompts = [
            "Once upon a time, there is a",
            "In far far awary, there is a",
            ]

    max_seq_len=256,
    max_batch_size=len(prompts),
    device = 'cuda'

    params_path=f'{llama_model}/params.json'
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        **params
    )

    #atten = SelfAttention(model_args)
    #atten.half()

    #x_shape = (2,1,512)

    #np.random.seed(0)
    #npx = np.random.randn(*x_shape)
    #x = ndl.Tensor(npx, dtype=ndl.fp16, backend=ndl.cuda)

    #start_pos = 5
    #norm_x = x.rms_norm()
    #result = atten.forward(norm_x, start_pos)
    #print('===== self attention =======')

    #fd = FeedForward(model_args)
    #fd.half()
    #result = fd(x)
    #print('======= feed forward =====')

    #model_args.vocab_size = 32000
    #encoder = EncoderBlock(model_args)
    #encoder.half()
    #result = encoder.forward(x, start_pos)
    #print('======= encoder =====')

    model_args.vocab_size = 32000
    transformer = Transformer(model_args)

    checkpoint_path= 'story_model/stories42M.pth'
    dtype = ndl.fp16
    transformer = load_meta_model(checkpoint_path, transformer, 
                                  model_args, params, dtype=dtype)

    transformer.half()
    #np_tok = np.ones((2,1), dtype=np.float32)
    np_tok = np.random.randn(2,1)
    np_tok[0][0] = 1.
    np_tok[1][0] = 1.
    ndl_tok = ndl.Tensor(np_tok, dtype=ndl.fp32, backend=ndl.cuda)

    start_pos = 1
    result = transformer.forward(tokens=ndl_tok, start_pos=start_pos)
    print('======= transformer =====')



