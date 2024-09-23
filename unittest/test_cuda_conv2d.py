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


        # NHWC
        self.shape = (2, 3, 128, 256)
        #self.shape = (2, 3, 4, 3)

        # KKIO
        self.wshape = (3, 3, 256, 16)

        self.npx = np.random.randn(*self.shape)
        self.weight = np.random.randn(*self.wshape)
        self.bias = np.random.randn(self.wshape[-1])

    def tch_conv_reference(self):
        tch_x = torch.from_numpy(self.npx).to(torch.float32)
        tch_w = torch.from_numpy(self.weight).to(torch.float32)
        tch_bias = torch.from_numpy(self.bias).to(torch.float32)

        # NHWC -> NCHW
        tch_x = tch_x.permute(0,3,1,2)

        # KKIO -> OIKK
        tch_w = tch_w.permute(3,2,0,1)

        # run convolution
        out = tch_nn.functional.conv2d(tch_x, tch_w, tch_bias, stride=1, padding=1)

        # NCHW -> NHWC
        return out.permute(0,2,3,1).contiguous().numpy()

    def np_conv_im2col(self):
        pad_width = ((0, 0),(1, 1),(1, 1),(0, 0))
        np_x = np.pad(self.npx, pad_width=pad_width, mode='constant', constant_values=0)

        N,H,W,C_in = np_x.shape
        K,_,_,C_out = self.weight.shape
        Ns, Hs, Ws, Cs = np_x.strides

        inner_dim = K * K * C_in
        A = np.lib.stride_tricks.as_strided(np_x, 
                                shape = (N, H-K+1, W-K+1, K, K, C_in),
                                strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)

        out = A @ self.weight.reshape(-1, C_out)
        out += self.bias
        return out.reshape(N,H-K+1,W-K+1,C_out)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def ndl_conv_im2col(self, dtype):
        ndl_x = ndl.Tensor(self.npx, dtype=dtype, backend=ndl.cuda)

        ndl_conv_layer = ndl_nn.Conv2d(in_channels=3,
                                       out_channels=self.wshape[-1],
                                       kernel_size=3,
                                       stride=1,
                                       bias=True,
                                       dtype=ndl.fp32, device=ndl.cuda)

        ndl_conv_layer.set_params([self.weight, self.bias])
        if dtype==ndl.fp16:
            ndl_conv_layer.half()

        ndl_out = ndl_conv_layer(ndl_x)
        ndl_out = ndl_out.to_numpy()

        return ndl_out

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

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        tch_out = self.tch_conv_reference()
        np_out = self.np_conv_im2col()
        ndl_out = self.ndl_conv_im2col(dtype=ndl.fp32)

        diff = tch_out - ndl_out
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")


if __name__ == '__main__':
    unittest.main()

