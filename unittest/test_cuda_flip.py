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

        self.npx = np.arange(1, 25).reshape(3, 2, 4)
        self.faxes = (0,1)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    '''
    def test_add_fp16_cuda(self):
        # CUDA, fp16
        tch_out = self.tch_conv_reference()
        np_out = self.np_conv_im2col()
        ndl_out = self.ndl_conv_im2col(dtype=ndl.fp16)

        diff = tch_out - ndl_out
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-0:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")
    '''

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        tch_x = torch.from_numpy(self.npx).float()
        tch_out = torch.flip(tch_x, self.faxes).numpy()

        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        ndl_out = ndl_x.flip(self.faxes).to_numpy()

        diff = tch_out - ndl_out
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

