import unittest
import numpy as np
import torch
import torch.nn as tch_nn

import needle as ndl
import needle.nn as ndl_nn

class TchModle(tch_nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.linear1 = tch_nn.Linear(out_feat, in_feat)
        self.linear2 = tch_nn.Linear(in_feat, out_feat)
        self.linear3 = tch_nn.Linear(out_feat, 10)

    def set_params(self, 
                   weight1, bias1, 
                   weight2, bias2, 
                   weight3, bias3):

        self.linear1.weight.data = torch.from_numpy(weight1).T
        self.linear1.bias.data = torch.from_numpy(bias1)

        self.linear2.weight.data = torch.from_numpy(weight2).T
        self.linear2.bias.data = torch.from_numpy(bias2)

        self.linear3.weight.data = torch.from_numpy(weight3).T
        self.linear3.bias.data = torch.from_numpy(bias3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        y = self.linear3(x)

        return y


class NdlModle(ndl_nn.Module):
    def __init__(self, in_feat, out_feat):
        super(NdlModle, self).__init__()

        # in_feat = 256
        # out_feat = 128
        self.linear1 = ndl_nn.Linear(out_feat, in_feat)
        self.linear2 = ndl_nn.Linear(in_feat, out_feat)
        self.linear3 = ndl_nn.Linear(out_feat, 10)
        #self.a = 6
        #self.model = ndl_nn.Sequential([self.linear1, self.linear2, self.linear3])

    def set_params(self, weight1, bias1, 
                   weight2, bias2, 
                   weight3, bias3):

        self.linear1.set_params([weight1, bias1]) # x.shape = (256, 128); weight1.shape = (128,256)
        self.linear2.set_params([weight2, bias2]) # x.shape = (256, 256); weight2.shape = (256, 128)
        self.linear3.set_params([weight3, bias3]) # x.shape = (256, 128); weight3.shape = (128, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        y = self.linear3(x)

        return y

class TestTensor(unittest.TestCase):
    def setUp(self):
        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"

        shape = (256,128)
        self.ndl_model = NdlModle(*shape)
        self.tch_model = TchModle(*shape)

        self.shape_weight1 = (128,256)
        self.shape_weight2 = (256,128)
        self.shape_weight3 = (128,10)

        self.set_params()
        self.set_ndl_params()
        self.set_tch_params()

        self.npx = np.random.randn(*shape).astype(np.float32)

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def set_params(self):
        self.weight1= np.random.randn(*self.shape_weight1).astype(np.float32)
        self.bias1= np.random.randn(self.shape_weight1[1]).astype(np.float32)

        self.weight2= np.random.randn(*self.shape_weight2).astype(np.float32)
        self.bias2= np.random.randn(self.shape_weight2[1]).astype(np.float32)

        self.weight3 = np.random.randn(*self.shape_weight3).astype(np.float32)
        self.bias3 = np.random.randn(self.shape_weight3[1]).astype(np.float32)

    def set_ndl_params(self):
        self.ndl_model.set_params(self.weight1, self.bias1,
                                  self.weight2, self.bias2, 
                                  self.weight3, self.bias3)

    def set_tch_params(self):
        self.tch_model.set_params(self.weight1, self.bias1, 
                                  self.weight2, self.bias2, 
                                  self.weight3, self.bias3)

    def ndl_forward(self, x):
        self.ndl_model.eval()
        #self.ndl_model.half()
        return self.ndl_model(x)

    def tch_forward(self, x):
        self.tch_model.eval()
        self.tch_model.cuda()
        #self.tch_model.cuda().half()
        return self.tch_model(x)

    '''
    def test_fp16_cuda(self):
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp16, backend=ndl.cuda)
        tch_x = torch.from_numpy(self.npx).to(torch.float16).cuda()

        ndl_y = self.ndl_forward(ndl_x).to_numpy()
        tch_y = self.tch_forward(tch_x).cpu().detach().numpy()

        self.check_shape(tch_y, ndl_y)
        err = max(abs(ndl_y - tch_y))

        if err < 1e-0:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}")
    '''

    def test_fp32_cuda(self):
        #self.tch_x = torch.from_numpy(self.npx).cuda()
        ndl_x = ndl.Tensor(self.npx, dtype=ndl.fp32, backend=ndl.cuda)
        tch_x = torch.from_numpy(self.npx).to(torch.float32).cuda()

        ndl_y = self.ndl_forward(ndl_x).to_numpy()
        tch_y = self.tch_forward(tch_x).cpu().detach().numpy()

        self.check_shape(tch_y, ndl_y)
        err = np.max(abs(ndl_y - tch_y))

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}")

if __name__ == '__main__':
    unittest.main()

