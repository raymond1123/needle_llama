import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

class TestTensor(unittest.TestCase):
    def setUp(self):
        #self.shape = (128, 256)
        np.random.seed(42)
        #self.shape = (8, 6)
        self.shape1 = (128, 1024)
        self.shape2 = (1024,512)
        self.npx1 = np.random.randn(*self.shape1)
        self.npx2 = np.random.randn(*self.shape2)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    '''
    def test_tensor_creation_fp32_cpu(self):
        # cpu, fp32
        tensor1 = ndl.Tensor(self.npx1, dtype=ndl.fp32, backend=ndl.cpu)
        tensor2 = ndl.Tensor(self.npx2, dtype=ndl.fp32, backend=ndl.cpu)
        self.check_shape(tensor1, self.npx1)
        self.check_shape(tensor2, self.npx2)
        self.assertIsNotNone(tensor1)
        self.assertIsNotNone(tensor2)

        print(f'fp32 err1={np.linalg.norm(tensor1.to_numpy()-self.npx1)}')
        print(f'fp32 err2={np.linalg.norm(tensor2.to_numpy()-self.npx2)}')
    '''

    def test_tensor_creation_fp16_cuda(self):
        # CUDA, fp16
        tensor1 = ndl.Tensor(self.npx1, dtype=ndl.fp16, backend=ndl.cuda)
        tensor2 = ndl.Tensor(self.npx2, dtype=ndl.fp16, backend=ndl.cuda)
        self.assertIsNotNone(tensor1)
        self.assertIsNotNone(tensor2)

        print(f'fp16 err={np.linalg.norm(tensor1.to_numpy()-self.npx1)}')
        print(f'fp16 err={np.linalg.norm(tensor2.to_numpy()-self.npx2)}')
        self.check_shape(tensor1, self.npx1)
        self.check_shape(tensor2, self.npx2)

    def test_tensor_creation_fp32_cuda(self):
        # CUDA, fp32
        tensor1 = ndl.Tensor(self.npx1, dtype=ndl.fp32, backend=ndl.cuda)
        tensor2 = ndl.Tensor(self.npx2, dtype=ndl.fp32, backend=ndl.cuda)
        self.assertIsNotNone(tensor1)
        self.assertIsNotNone(tensor2)

        print(f'fp32 err={np.linalg.norm(tensor1.to_numpy()-self.npx1)}')
        print(f'fp32 err={np.linalg.norm(tensor2.to_numpy()-self.npx2)}')
        self.check_shape(tensor1, self.npx1)
        self.check_shape(tensor2, self.npx2)

if __name__ == '__main__':
    unittest.main()

