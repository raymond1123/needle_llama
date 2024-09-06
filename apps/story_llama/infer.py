from typing import Optional
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer
import numpy as np
from load_params import load_meta_model
import needle as ndl
import needle.nn as nn


#### debug
import torch
#### debug done


class LLaMA:
    def __init__(self, model: Transformer, 
                 tokenizer: SentencePieceProcessor, 
                 model_args: ModelArgs):

        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoint_path: str, 
              tokenizer_path: str, 
              params_path: str,
              max_seq_len: int, 
              max_batch_size: int, 
              device: str,
              dtype):

        with open(params_path, "r") as f:
            params = json.loads(f.read())

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        model_args.vocab_size = tokenizer.vocab_size()
        model = Transformer(model_args)
        model = load_meta_model(checkpoint_path, model, 
                                model_args, params, dtype=dtype)
        model.half()

        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts, temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = ndl.fill_val([batch_size, total_len], float(pad_id),
                              dtype=ndl.fp32, device=ndl.cuda)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k:k+1, : len(t)] = ndl.Tensor(np.array(t).astype(np.float32), 
                                             dtype=ndl.fp32, backend=ndl.cuda)

        #eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            cal_tokens = tokens[:, cur_pos-1:cur_pos]
            logits = self.model.forward(cal_tokens, cur_pos)
            logits_shape = [logits.shape[0], logits.shape[-1]]

            if temperature > 0:
                # The temperature is applied before the softmax
                logits = (logits / temperature).reshape(logits_shape)
                probs = logits.softmax()

                next_token = self._sample_top_p(probs, top_p)
            else:
                pass
                # Greedily select the token with the max probability

            ## temporary, only greedy
            next_token, max_val = logits.reshape(logits_shape).argmax(dim=-1)
            # Only replace token if it is a padding token
            prompt_mask = prompt_tokens_mask[:, cur_pos:cur_pos+1].reshape(list(next_token.shape))
            selected_tokens = tokens[:, cur_pos:cur_pos+1].reshape(list(next_token.shape))
            next_token = ndl.where(prompt_mask,
                                   selected_tokens,
                                   next_token)

            tokens[:, cur_pos:cur_pos+1] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            '''
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break
            '''
        #print()

        out_tokens = []
        out_text = []
        tokens = tokens.to_numpy().tolist()
        tokens = [[int(x) for x in sublist] for sublist in tokens]

        for prompt_index, current_prompt_tokens in enumerate(tokens):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token


if __name__ == '__main__':
    llama_model = './story_model'
    device = 'cuda'

    prompts = [
            "Once upon a time, there is a",
            "In far far away, there is a",
            ]

    model = LLaMA.build(
        checkpoint_path=f'{llama_model}/stories42M.pth',
        tokenizer_path=f'{llama_model}/tokenizer.model',
        params_path=f'{llama_model}/params.json',
        max_seq_len=512,
        max_batch_size=len(prompts),
        device=device,
        dtype=ndl.fp16
    )

    out_tokens, out_texts = (model.text_completion(prompts, temperature=0, max_gen_len=1024))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)

