import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

class TestTensor(unittest.TestCase):
    def setUp(self):
        #self.shape = (128, 256)

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"


        self.scalar = 0.5
        self.shape = (128, 256)
        self.npx1 = np.random.randn(*self.shape)
        self.npx2 = np.random.randn(*self.shape)
        self.npx = self.npx1 + self.npx2
        self.npx_scalar = self.npx1+self.scalar

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def eadd(self, dtype, backend):
        tensor1 = ndl.Tensor(self.npx1, dtype=dtype, backend=backend)
        tensor2 = ndl.Tensor(self.npx2, dtype=dtype, backend=backend)
        tensor1 += tensor2

        return tensor1

    def add(self, dtype, backend):
        tensor1 = ndl.Tensor(self.npx1, dtype=dtype, backend=backend)
        tensor2 = ndl.Tensor(self.npx2, dtype=dtype, backend=backend)
        tensor = tensor1 + tensor2
        self.assertIsNotNone(tensor)

        return tensor

    def add_scalar(self, dtype, backend):
        tensor = ndl.Tensor(self.npx1, dtype=dtype, backend=backend)
        tensor = tensor + self.scalar
        self.assertIsNotNone(tensor)

        return tensor

    def add_scalar_r(self, dtype, backend):
        tensor = ndl.Tensor(self.npx1, dtype=dtype, backend=backend)
        tensor = self.scalar + tensor
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

    def test_add_fp16_cuda(self):
        # CUDA, fp16
        tensor = self.add(ndl.fp16, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx)
        self.check_shape(tensor, self.npx)

        if err < 1e-1:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        tensor = self.add(ndl.fp32, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx)
        self.check_shape(tensor, self.npx)

        if err < 1e-5:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_scalar_fp16_cuda(self):
        # CUDA, fp16
        tensor = self.add_scalar(ndl.fp16, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx_scalar)
        self.check_shape(tensor, self.npx)

        if err < 1e-1:
            print(f"fp16_cuda_scalar: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda_scalar: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_scalar_fp32_cuda(self):
        # CUDA, fp32
        tensor = self.add_scalar(ndl.fp32, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx_scalar)
        self.check_shape(tensor, self.npx)

        if err < 1e-5:
            print(f"fp32_cuda_scalar: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda_scalar: {self.RED}FAILED{self.RESET}, {err=}")

    def test_addr_scalar_fp16_cuda(self):
        # CUDA, fp16
        tensor = self.add_scalar_r(ndl.fp16, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx_scalar)
        self.check_shape(tensor, self.npx)

        if err < 1e-1:
            print(f"fp16_cuda_scalar reverse: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda_scalar reverse: {self.RED}FAILED{self.RESET}, {err=}")

    def test_addr_scalar_fp32_cuda(self):
        # CUDA, fp32
        tensor = self.add_scalar_r(ndl.fp32, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx_scalar)
        self.check_shape(tensor, self.npx)

        if err < 1e-5:
            print(f"fp32_cuda_scalar reverse: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda_scalar reverse: {self.RED}FAILED{self.RESET}, {err=}")

    def test_eadd_fp16_cuda(self):
        # CUDA, fp16
        tensor = self.eadd(ndl.fp16, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx)
        self.check_shape(tensor, self.npx)

        if err < 1e-1:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        tensor = self.eadd(ndl.fp32, ndl.cuda)
        err=np.linalg.norm(tensor.to_numpy()-self.npx)
        self.check_shape(tensor, self.npx)

        if err < 1e-5:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

