import unittest
import numpy as np
import needle as ndl

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        self.shape = (5, 128, 256)
        self.npx1 = np.random.randn(*self.shape)
        self.npx2 = np.random.randn(*self.shape)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_fp16_zero_cuda(self):
        # CUDA, fp16
        ndl_x1 = ndl.Tensor(self.npx1, dtype=ndl.fp16, backend=ndl.cuda)
        ndl_x2 = ndl.Tensor(self.npx2, dtype=ndl.fp16, backend=ndl.cuda)

        ndl_x1[1:,3:,:] = ndl_x2[1:,3:,:]
        self.npx1[1:,3:,:] = self.npx2[1:,3:,:]

        ndl_x1 = ndl_x1.to_numpy()
        err=np.linalg.norm(ndl_x1 - self.npx1)
        print(f'{err=}')

        if err < 1e-1:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp32_zero_cuda(self):
        # CUDA, fp32
        ndl_x1 = ndl.Tensor(self.npx1, dtype=ndl.fp32, backend=ndl.cuda)
        ndl_x2 = ndl.Tensor(self.npx2, dtype=ndl.fp32, backend=ndl.cuda)

        ndl_x1[1:,3:,:] = ndl_x2[1:,3:,:]
        self.npx1[1:,3:,:] = self.npx2[1:,3:,:]

        ndl_x1 = ndl_x1.to_numpy()
        err=np.linalg.norm(ndl_x1 - self.npx1)

        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

