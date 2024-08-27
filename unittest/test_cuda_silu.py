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


        self.shape = (2, 3, 128, 256)
        self.npx = np.random.randn(*self.shape)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_add_fp16_cuda(self):
        # CUDA, fp16
        self.ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        self.tch_x = torch.from_numpy(self.npx).to(torch.float16).cuda()

        tch_scores = tch_F.silu(self.tch_x, dim=-1).cpu().detach().numpy()
        ndl_scores = self.ndl_x.silu().to_numpy()
        #self.check_shape(tensor, self.npx)

        diff = tch_scores - ndl_scores
        #err=np.linalg.norm(ndl_scores-tch_scores)
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        self.ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        self.tch_x = torch.from_numpy(self.npx).to(torch.float32).cuda()

        tch_scores = tch_F.silu(self.tch_x, dim=-1).cpu().detach().numpy()
        ndl_scores = self.ndl_x.silu().to_numpy()
        #self.check_shape(tensor, self.npx)

        diff = tch_scores - ndl_scores
        err=np.linalg.norm(ndl_scores-tch_scores)
        #err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

