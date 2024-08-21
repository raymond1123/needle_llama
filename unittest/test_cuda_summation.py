import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"


        self.shape = (2, 3, 128, 256)
        self.axes = (1,3)
        self.npx = np.random.randn(*self.shape)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def add(self, dtype, backend):
        tensor = ndl.Tensor(self.npx, dtype=dtype, backend=backend)
        tensor = tensor.summation(self.axes)
        self.npx = np.sum(self.npx, axis=self.axes)
        self.assertIsNotNone(tensor)

        return tensor

    def test_add_fp16_cuda(self):
        # CUDA, fp16
        tensor = self.add(ndl.fp16, ndl.cuda)
        #err=np.linalg.norm(tensor.to_numpy()-self.npx)
        self.check_shape(tensor, self.npx)

        diff = tensor.to_numpy() - self.npx
        err = np.max(np.abs(diff))

        if err < 1e-0:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        tensor = self.add(ndl.fp32, ndl.cuda)
        #err=np.linalg.norm(tensor.to_numpy()-self.npx)
        self.check_shape(tensor, self.npx)

        diff = tensor.to_numpy() - self.npx
        err = np.max(np.abs(diff))

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

