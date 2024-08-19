import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

class TestTensor(unittest.TestCase):
    def setUp(self):
        #self.shape = (128, 256)

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        #self.shape1 = (3, 2)
        #self.shape2 = (2, 3)
        self.shape1 = (1024, 512)
        self.shape2 = (512, 256)
        #self.shape1 = (128, 512)
        #self.shape2 = (512, 128)

        self.npx1 = np.random.randn(*self.shape1)
        self.npx2 = np.random.randn(*self.shape2)
        self.npx = self.npx1@self.npx2

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def matmul(self, dtype, backend):
        tensor1 = ndl.Tensor(self.npx1, dtype=dtype, backend=backend)
        tensor2 = ndl.Tensor(self.npx2, dtype=dtype, backend=backend)
        tensor = tensor1@tensor2
        self.assertIsNotNone(tensor)

        return tensor

    def test_matmul_fp16_cuda(self):
        # CUDA, fp16
        tensor = self.matmul(ndl.fp16, ndl.cuda)
        diff = tensor.to_numpy() - self.npx
        err = np.max(np.abs(diff))
        print(f'{err=}')
        self.check_shape(tensor, self.npx)

        if err < 1e-0:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}")

    def test_matmul_fp32_cuda(self):
        # CUDA, fp32
        tensor = self.matmul(ndl.fp32, ndl.cuda)
        diff = tensor.to_numpy() - self.npx
        err = np.max(np.abs(diff))
        print(f'{err=}')
        self.check_shape(tensor, self.npx)

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}")

if __name__ == '__main__':
    unittest.main()

