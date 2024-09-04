import unittest
import numpy as np
import needle as ndl

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        self.shape = (128, 256)
        self.npx = np.zeros(self.shape)
        self.npx_one = np.ones(self.shape)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_fp16_zero_cuda(self):
        # CUDA, fp16
        ndl_x = ndl.zeros(self.shape, dtype=ndl.fp16, device=ndl.cuda)

        err=np.linalg.norm(ndl_x.to_numpy())
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp32_zero_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.zeros(self.shape, dtype=ndl.fp32, device=ndl.cuda)
        err=np.linalg.norm(ndl_x.to_numpy())

        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp16_one_cuda(self):
        # CUDA, fp16
        ndl_x = ndl.ones(self.shape, dtype=ndl.fp16, device=ndl.cuda)

        err=np.linalg.norm(ndl_x.to_numpy()-self.npx_one)
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp32_ones_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.ones(self.shape, dtype=ndl.fp32, device=ndl.cuda)
        err=np.linalg.norm(ndl_x.to_numpy()-self.npx_one)

        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp16_fillvalue_cuda(self):
        # CUDA, fp16
        npx_v = np.full(self.shape, -1)
        ndl_x = ndl.fill_val(list(self.shape), -1., dtype=ndl.fp16, device=ndl.cuda)

        err=np.linalg.norm(ndl_x.to_numpy()-npx_v)
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp32_fillvalue_cuda(self):
        # CUDA, fp32
        npx_v = np.full(self.shape, -1)
        ndl_x = ndl.fill_val(list(self.shape), -1., dtype=ndl.fp32, device=ndl.cuda)

        err=np.linalg.norm(ndl_x.to_numpy()-npx_v)

        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

