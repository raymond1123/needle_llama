import unittest
import numpy as np
import needle as ndl
import needle.nn as ndl_nn

import torch
import torch.nn as tch_nn

class TorchRMSNorm(tch_nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # The gamma parameter
        self.weight = tch_nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        self.shape = (3, 128, 256)
        self.npx = np.random.randn(*self.shape)

        self.tch_rms = TorchRMSNorm(self.shape[-1])

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_add_fp16_cuda(self):
        # CUDA, fp16
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        tch_x = torch.from_numpy(self.npx).to(torch.float16).cuda()

        self.tch_rms.cuda().half()
        tch_result = self.tch_rms(tch_x).cpu().detach().numpy()

        ndl_result = ndl_x.rms_norm()
        err = np.max(np.abs(ndl_result.to_numpy()-tch_result))
        print(f'{err=}')


        if err < 1e-1:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        tch_x = torch.from_numpy(self.npx).to(torch.float32).cuda()

        self.tch_rms.cuda()
        tch_result = self.tch_rms(tch_x).cpu().detach().numpy()

        ndl_x.rms_norm()
        ndl_result = ndl_x.rms_norm().to_numpy()

        err = np.max(np.abs(ndl_result-tch_result))
        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()


