import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

class TestTensor(unittest.TestCase):
    def setUp(self):
        #self.shape = (128, 256)
        np.random.seed(42)
        self.shape = (8, 6)
        self.npx = np.random.randn(*self.shape)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_tensor_creation_fp32_cpu(self):
        # cpu, fp32
        tensor = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cpu)
        self.check_shape(tensor, self.npx)

        print(f'fp32 err={np.linalg.norm(tensor.to_numpy()-self.npx)}')
        self.assertIsNotNone(tensor)

    def test_tensor_creation_fp16_cuda(self):
        # CUDA, fp16
        print(self.npx)
        tensor = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        self.assertIsNotNone(tensor)

        print(f'fp16 err={np.linalg.norm(tensor.to_numpy()-self.npx)}')
        self.check_shape(tensor, self.npx)

    def test_tensor_creation_fp32_cuda(self):
        # CUDA, fp32
        tensor = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        self.assertIsNotNone(tensor)

        print(f'fp32 err={np.linalg.norm(tensor.to_numpy()-self.npx)}')
        self.check_shape(tensor, self.npx)

if __name__ == '__main__':
    unittest.main()

