import unittest
import numpy as np
import needle as ndl
import needle.nn as ndl_nn

import torch 
import torch.nn as tch_nn

class TestTensor(unittest.TestCase):
    def setUp(self):

        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        np.random.seed(0)
        self.npx = np.random.randint(1, 10, size=(2, 5)).astype(np.float32)
        self.axes = (0,1)
        self.dilation = 1

    import numpy as np

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_add_fp16_cuda(self):
        # CUDA, fp16
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        ndl_out = ndl_x.dilate(self.dilation, self.axes).to_numpy()

        result = np.array([[6., 0., 1., 0., 4., 0., 4., 0., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [4., 0., 6., 0., 3., 0., 5., 0., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        diff = result - ndl_out
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        ndl_out = ndl_x.dilate(self.dilation, self.axes).to_numpy()

        result = np.array([[6., 0., 1., 0., 4., 0., 4., 0., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [4., 0., 6., 0., 3., 0., 5., 0., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        diff = result - ndl_out
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

