import unittest
import numpy as np
import needle as ndl
import needle.nn as ndl_nn

import torch
import torch.nn as tch_nn

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        self.shape = (3, 128, 256)
        self.npx = np.random.randn(*self.shape)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_add_fp16_cuda(self):
        # CUDA, fp16
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        tch_x = torch.from_numpy(self.npx)

        tch_val, tch_idx = torch.max(tch_x, dim=2)
        ndl_val, ndl_idx = ndl_x.argmax(dim=2)

        err_val = np.max(np.abs(ndl_val.to_numpy() - tch_val.numpy()))
        err_idx = np.max(np.abs(ndl_idx.to_numpy() - tch_idx.numpy()))
        print(f'{err_val=}, {err_idx=}')

        if err_val < 1e-2 and err_idx < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        tch_x = torch.from_numpy(self.npx)

        tch_val, tch_idx = torch.max(tch_x, dim=2)
        ndl_val, ndl_idx = ndl_x.argmax(dim=2)

        err_val = np.max(np.abs(ndl_val.to_numpy() - tch_val.numpy()))
        err_idx = np.max(np.abs(ndl_idx.to_numpy() - tch_idx.numpy()))
        print(f'{err_val=}, {err_idx=}')

        if err_val < 1e-2 and err_idx < 1e-2:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()


