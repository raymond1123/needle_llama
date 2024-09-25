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
        #self.shape = (2, 128, 128, 256)
        self.shape = (2, 4, 4, 3)

        # KKIO
        #self.wshape = (3, 3, 256, 16)
        self.wshape = (3, 3, 3, 2)

        self.npx = np.random.randn(*self.shape)
        self.weight = np.random.randn(*self.wshape)
        self.bias = np.random.randn(self.wshape[-1])

    def tch_conv_reference(self):
        self.tch_x = torch.from_numpy(self.npx).to(torch.float32).requires_grad_(True)
        self.tch_w = torch.from_numpy(self.weight).to(torch.float32).requires_grad_(True)
        self.tch_bias = torch.from_numpy(self.bias).to(torch.float32).requires_grad_(True)

        # NHWC -> NCHW
        self.tch_x = self.tch_x.permute(0,3,1,2)
        self.tch_x.retain_grad()  # 保留梯度信息

        # KKIO -> OIKK
        self.tch_w = self.tch_w.permute(3,2,0,1)
        self.tch_w.retain_grad()  # 保留梯度信息

        # run convolution
        out = tch_nn.functional.conv2d(self.tch_x, self.tch_w, self.tch_bias, stride=1, padding=1)

        # NCHW -> NHWC
        return out.permute(0,2,3,1).contiguous()

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
        self.ndl_x = ndl.Tensor(self.npx, dtype=dtype, backend=ndl.cuda)

        ndl_conv_layer = ndl_nn.Conv2d(in_channels=3,
                                       out_channels=self.wshape[-1],
                                       kernel_size=3,
                                       stride=1,
                                       bias=True,
                                       dtype=ndl.fp32, device=ndl.cuda)

        ndl_conv_layer.set_params([self.weight, self.bias])
        if dtype==ndl.fp16:
            ndl_conv_layer.half()

        ndl_out = ndl_conv_layer(self.ndl_x)

        self.ndl_w = ndl_conv_layer.weight
        self.ndl_b = ndl_conv_layer.bias

        return ndl_out

    def test_add_fp16_cuda(self):
        # CUDA, fp16
        tch_out = self.tch_conv_reference().detach().numpy()
        np_out = self.np_conv_im2col()
        ndl_out = self.ndl_conv_im2col(dtype=ndl.fp16).to_numpy()

        diff = tch_out - ndl_out
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-0:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_add_fp32_cuda(self):
        # CUDA, fp32
        tch_out = self.tch_conv_reference().detach().numpy()
        np_out = self.np_conv_im2col()
        ndl_out = self.ndl_conv_im2col(dtype=ndl.fp32).to_numpy()

        diff = tch_out - ndl_out
        err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp32_backprop_cuda(self):
        # CUDA, fp32
        tch_out = torch.sum(self.tch_conv_reference())
        ndl_out = self.ndl_conv_im2col(dtype=ndl.fp32).summation()

        tch_out.backward()
        tch_x_grad = self.tch_x.grad.permute(0,2,3,1).cpu().numpy()
        tch_w_grad = self.tch_w.grad.cpu().permute(2,3,1,0).numpy()
        tch_b_grad = self.tch_bias.grad.cpu().numpy()

        ndl_out.backward()
        ndl_x_grad = self.ndl_x.grad()
        ndl_b_grad = self.ndl_b.grad()
        ndl_w_grad = self.ndl_w.grad()

        diff_x = ndl_x_grad - tch_x_grad
        diff_w = ndl_w_grad - tch_w_grad
        diff_b = ndl_b_grad - tch_b_grad

        err_x = np.max(np.abs(diff_x))
        err_w = np.max(np.abs(diff_w))
        err_b = np.max(np.abs(diff_b))

        print(f'{err_x=}, {err_w=}, {err_b}')

        if err_x < 1e-5 and err_w < 1e-5 and err_b < 1e-5:
            print(f"backward fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"backward fp32_cuda: {self.RED}FAILED{self.RESET}")

    def test_fp16_backprop_cuda(self):
        # CUDA, fp32
        tch_out = torch.sum(self.tch_conv_reference())
        ndl_out = self.ndl_conv_im2col(dtype=ndl.fp16).summation()

        tch_out.backward()
        tch_x_grad = self.tch_x.grad.permute(0,2,3,1).cpu().numpy()
        tch_w_grad = self.tch_w.grad.cpu().permute(2,3,1,0).numpy()
        tch_b_grad = self.tch_bias.grad.cpu().numpy()

        ndl_out.backward()
        ndl_x_grad = self.ndl_x.grad()
        ndl_b_grad = self.ndl_b.grad()
        ndl_w_grad = self.ndl_w.grad()

        diff_x = ndl_x_grad - tch_x_grad
        diff_w = ndl_w_grad - tch_w_grad
        diff_b = ndl_b_grad - tch_b_grad

        err_x = np.max(np.abs(diff_x))
        err_w = np.max(np.abs(diff_w))
        err_b = np.max(np.abs(diff_b))
        print(f'{err_x=}, {err_w=}, {err_b}')

        if err_x < 1e-1 and err_w < 1e-1 and err_b < 1e-1:
            print(f"backward fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"backward fp16_cuda: {self.RED}FAILED{self.RESET}")

if __name__ == '__main__':
    unittest.main()

