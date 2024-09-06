import json
from pathlib import Path

import numpy as np

import torch
import torch.nn as tch_nn

import needle as ndl
import needle.nn as nn


def load_meta_model(model_path, transformer, config, params, dtype):

    model = torch.load(model_path, map_location='cpu')
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    #state_dict = {}
    state_dict = model['model']

    # set ModelArgs
    config.dim = params["dim"]
    config.n_layers = params["n_layers"]
    config.n_heads = params["n_heads"]
    config.n_kv_heads = params.get('n_kv_heads') or params['n_heads']
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]

    config.vocab_size = state_dict['tok_embeddings.weight'].shape[0]
    #config.max_seq_len = 1024 

    tok_embeddings = state_dict['tok_embeddings.weight'].numpy().astype(np.float32)
    norm_weight = state_dict['norm.weight'].numpy().astype(np.float32)

    transformer.rms_weight = ndl.Tensor(norm_weight, dtype=dtype, backend=ndl.cuda)
    transformer.tok_embeddings.set_params(tok_embeddings, dtype=dtype, device=ndl.cuda)

    if dtype==ndl.fp16:
        transformer.half()

    for i, layer in enumerate(transformer.layers):
        x_rms_weight = state_dict[f'layers.{i}.attention_norm.weight'].numpy()
        h_rms_weight = state_dict[f'layers.{i}.ffn_norm.weight'].numpy()

        layer.x_rms_weight = ndl.Tensor(x_rms_weight, dtype=dtype, backend=ndl.cuda)
        layer.h_rms_weight = ndl.Tensor(h_rms_weight, dtype=dtype, backend=ndl.cuda)

        attention_wq = state_dict[f'layers.{i}.attention.wq.weight'].numpy().T
        attention_wk = state_dict[f'layers.{i}.attention.wk.weight'].numpy().T
        attention_wv = state_dict[f'layers.{i}.attention.wv.weight'].numpy().T
        attention_wo = state_dict[f'layers.{i}.attention.wo.weight'].numpy().T

        layer.attention.wq.set_params([attention_wq], dtype=dtype, device=ndl.cuda)
        layer.attention.wk.set_params([attention_wk], dtype=dtype, device=ndl.cuda)
        layer.attention.wv.set_params([attention_wv], dtype=dtype, device=ndl.cuda)
        layer.attention.wo.set_params([attention_wo], dtype=dtype, device=ndl.cuda)


        feed_forward_w1 = state_dict[f'layers.{i}.feed_forward.w1.weight'].numpy().T
        feed_forward_w2 = state_dict[f'layers.{i}.feed_forward.w2.weight'].numpy().T
        feed_forward_w3 = state_dict[f'layers.{i}.feed_forward.w3.weight'].numpy().T
        layer.feed_forward.w1.set_params([feed_forward_w1], dtype=dtype, device=ndl.cuda)
        layer.feed_forward.w2.set_params([feed_forward_w2], dtype=dtype, device=ndl.cuda)
        layer.feed_forward.w3.set_params([feed_forward_w3], dtype=dtype, device=ndl.cuda)

    # final classifier
    output_weight = state_dict['output.weight'].numpy().T
    transformer.output.set_params([output_weight], dtype=dtype, device=ndl.cuda)
    return transformer


if __name__ == "__main__":

    from my_model import Transformer, ModelArgs

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

    model_args.vocab_size = 32000
    transformer = Transformer(model_args)

    model_path = 'story_model/stories42M.pth'
    dtype = ndl.fp16
    transformer = load_meta_model(model_path, transformer, params, dtype=dtype)

