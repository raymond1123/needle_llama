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
        self.npy = np.random.randn(*self.shape)

        self.condition = self.npx < 0.5
        self.ndl_condition = ndl.Tensor(self.condition, dtype=ndl.fp32, backend=ndl.cuda)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_add_fp16_cuda(self):
        # CUDA, fp16

        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        ndl_y = ndl.Tensor(self.npy, dtype=ndl.fp16, backend=ndl.cuda)

        tch_x = torch.from_numpy(self.npx)
        tch_y = torch.from_numpy(self.npy)
        tch_condition = tch_x<0.5

        tch_result = torch.where(tch_condition, tch_x, tch_y)
        tch_result = tch_result.numpy().astype(np.float32)

        ndl_result = ndl.where(self.ndl_condition, ndl_x, ndl_y)
        ndl_result = ndl_result.to_numpy()

        err = np.max(np.abs(ndl_result-tch_result))
        print(f'{err=}')

        if err < 1e-1:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        ndl_y = ndl.Tensor(self.npy, dtype=ndl.fp32, backend=ndl.cuda)

        tch_x = torch.from_numpy(self.npx)
        tch_y = torch.from_numpy(self.npy)
        tch_condition = tch_x<0.5

        tch_result = torch.where(tch_condition, tch_x, tch_y)
        tch_result = tch_result.numpy().astype(np.float32)

        ndl_result = ndl.where(self.ndl_condition, ndl_x, ndl_y)
        ndl_result = ndl_result.to_numpy()

        err = np.max(np.abs(ndl_result-tch_result))
        print(f'{err=}')

        if err < 1e-1:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()


