import unittest
import numpy as np
import needle as ndl  # 导入你封装的 C++ 模块

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        self.shape = (4,4)
        self.npx = np.arange(16, dtype=np.float32).reshape(self.shape)

        self.window_shape = (2, 2)
        self.new_shape = (self.npx.shape[0] - self.window_shape[0] + 1,
                          self.npx.shape[1] - self.window_shape[1] + 1) + self.window_shape
        self.np_new_strides = self.npx.strides * 2

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        np_result = np.lib.stride_tricks.as_strided(self.npx, 
                                                    shape=self.new_shape, 
                                                    strides=self.np_new_strides)

        np_fp32_size = np.dtype(np.float32).itemsize
        ndl_new_strides = [x//np_fp32_size for x in list(self.np_new_strides)]
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        ndl_result = ndl_x.as_strided(self.new_shape, ndl_new_strides)

        ndl_result = ndl_result.to_numpy()

        diff = np_result - ndl_result
        err=np.linalg.norm(diff)
        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

