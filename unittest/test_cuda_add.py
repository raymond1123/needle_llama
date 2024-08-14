import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

class TestTensor(unittest.TestCase):
    def setUp(self):
        #self.shape = (128, 256)
        self.shape = (3, 2)
        self.npx1 = np.random.randn(*self.shape)
        self.npx2 = np.random.randn(*self.shape)
        self.npx = self.npx1 + self.npx2

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def add(self, dtype, backend):
        tensor1 = ndl.Tensor(self.npx1, dtype=dtype, backend=backend)
        tensor2 = ndl.Tensor(self.npx2, dtype=dtype, backend=backend)
        tensor = tensor1 + tensor2
        self.assertIsNotNone(tensor)

        return tensor

    '''
    def test_tensor_creation_fp32_cpu(self):
        # cpu, fp32
        tensor = self.add(ndl.fp32, ndl.cpu)
        #tensor1 = ndl.Tensor(self.npx1, dtype=ndl.fp32, backend=ndl.cuda)
        #tensor2 = ndl.Tensor(self.npx2, dtype=ndl.fp32, backend=ndl.cuda)
        #tensor = tensor1 + tensor2
        self.assertIsNotNone(tensor)
        print(f'fp32_cpu: err={np.linalg.norm(tensor.to_numpy()-self.npx)}')
        self.check_shape(tensor, self.npx)
    '''

    def test_tensor_creation_fp16_cuda(self):
        # CUDA, fp16
        tensor = self.add(ndl.fp16, ndl.cuda)
        print(f'fp16_cuda: err={np.linalg.norm(tensor.to_numpy()-self.npx)}')
        self.check_shape(tensor, self.npx)

    def test_tensor_creation_fp32_cuda(self):
        # CUDA, fp32
        tensor = self.add(ndl.fp32, ndl.cuda)
        print(f'fp32_cuda: err={np.linalg.norm(tensor.to_numpy()-self.npx)}')
        self.check_shape(tensor, self.npx)

if __name__ == '__main__':
    unittest.main()

