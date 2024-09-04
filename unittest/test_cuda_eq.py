import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

import torch 
import torch.nn as tch_nn
import torch.nn.functional as tch_F

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        self.shape = (256, 512)
        self.npx = np.random.randn(*self.shape)
        self.npy = self.npx
        self.npy[:,122:233] = self.npx[:, 122:233]+0.5

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_eq_fp16_cuda(self):
        # CUDA, fp16
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        ndl_y = ndl.Tensor(self.npy, dtype=ndl.fp16, backend=ndl.cuda)

        tch_x = torch.from_numpy(self.npx)
        tch_y = torch.from_numpy(self.npy)

        ndl_result = (ndl_x==ndl_y).to_numpy()
        tch_result = (tch_x==tch_y).numpy()

        diff = tch_result - ndl_result
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_eq_fp32_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        ndl_y = ndl.Tensor(self.npy, dtype=ndl.fp32, backend=ndl.cuda)

        tch_x = torch.from_numpy(self.npx)
        tch_y = torch.from_numpy(self.npy)

        ndl_result = (ndl_x==ndl_y).to_numpy()
        tch_result = (tch_x==tch_y).numpy()

        diff = tch_result - ndl_result
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_neq_fp16_cuda(self):
        # CUDA, fp16
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        ndl_y = ndl.Tensor(self.npy, dtype=ndl.fp16, backend=ndl.cuda)

        tch_x = torch.from_numpy(self.npx)
        tch_y = torch.from_numpy(self.npy)

        ndl_result = (ndl_x!=ndl_y).to_numpy()
        tch_result = (tch_x!=tch_y).numpy()

        diff = tch_result - ndl_result
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_neq_fp32_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        ndl_y = ndl.Tensor(self.npy, dtype=ndl.fp32, backend=ndl.cuda)

        tch_x = torch.from_numpy(self.npx)
        tch_y = torch.from_numpy(self.npy)

        ndl_result = (ndl_x!=ndl_y).to_numpy()
        tch_result = (tch_x!=tch_y).numpy()

        diff = tch_result - ndl_result
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

