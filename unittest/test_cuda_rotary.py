import unittest
import numpy as np
import needle as ndl
import needle.nn as ndl_nn

import torch
import torch.nn as tch_nn

def precompute_theta_pos_frequencies(head_dim: int, 
                                     seq_len: int, 
                                     device: str, 
                                     theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        self.device = 'cuda'

        self.dim = 512
        self.num_heads = 8
        self.head_dim = self.dim//self.num_heads
        self.seq_len = 1024
        self.shape = (self.num_heads, self.head_dim)

        self.npx = np.random.randn(*self.shape).astype(np.float32)
        self.ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        self.tch_x = torch.from_numpy(self.npx).to(torch.float32).cuda()

        self.ndl_x_fp16 = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_add_fp16_cuda(self):
        # CUDA, fp32
        freqs_complex = precompute_theta_pos_frequencies(self.head_dim,
                                                         self.seq_len*2, 
                                                         device=self.device)
        start = 5; seq_len=1
        tch_y = apply_rotary_embeddings(self.tch_x, freqs_complex[start:start+seq_len], self.device)
        tch_y = tch_y.cpu().detach().numpy()

        ndl_y = self.ndl_x_fp16.rotary_emb(start).to_numpy()
        err = np.max(np.abs(ndl_y-tch_y))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        freqs_complex = precompute_theta_pos_frequencies(self.head_dim,
                                                         self.seq_len*2, 
                                                         device=self.device)
        start = 5; seq_len=1
        tch_y = apply_rotary_embeddings(self.tch_x, freqs_complex[start:start+seq_len], self.device)
        tch_y = tch_y.cpu().detach().numpy()

        ndl_y = self.ndl_x.rotary_emb(start).to_numpy()
        err = np.max(np.abs(ndl_y-tch_y))
        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()


