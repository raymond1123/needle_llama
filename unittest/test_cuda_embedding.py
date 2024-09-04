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

        #self.shape = (32000, 512)
        #self.shape = (320, 512)
        self.shape = (32000, 512)
        self.idx = np.array([[0,1],[2,3],[4,2],[233,325]])
        #self.idx = np.array([[1],[1]])

    def check_shape(self, tensor, npx):
        self.assertEqual(tensor.shape, npx.shape)

    def test_fp16_cuda(self):
        # CUDA, fp16
        ndl_idx = ndl.Tensor(self.idx, dtype=ndl.fp32, backend=ndl.cuda)
        tch_idx = torch.from_numpy(self.idx).cuda()

        tch_emb = tch_nn.Embedding(*self.shape).to(torch.float16).cuda()
        np_emb = tch_emb.weight.cpu().detach().numpy()

        ndl_emb = ndl_nn.Embedding(*self.shape, dtype=ndl.fp16, device=ndl.cuda)
        ndl_emb.set_params(np_emb, dtype=ndl.fp16, device=ndl.cuda)

        tch_result = tch_emb(tch_idx).cpu().detach().numpy()
        ndl_result = ndl_emb(ndl_idx).to_numpy()

        diff = tch_result - ndl_result
        err=np.linalg.norm(diff)
        #err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-2:
            print(f"fp16_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp16_cuda: {self.RED}FAILED{self.RESET}, {err=}")

    def test_fp32_cuda(self):
        # CUDA, fp32

        ndl_idx = ndl.Tensor(self.idx, dtype=ndl.fp32, backend=ndl.cuda)
        tch_idx = torch.from_numpy(self.idx).cuda()


        tch_emb = tch_nn.Embedding(*self.shape).to(torch.float16).cuda()
        np_emb = tch_emb.weight.cpu().detach().numpy()

        ndl_emb = ndl_nn.Embedding(*self.shape, dtype=ndl.fp32, device=ndl.cuda)
        ndl_emb.set_params(np_emb, dtype=ndl.fp32, device=ndl.cuda)


        tch_result = tch_emb(tch_idx).cpu().detach().numpy()
        ndl_result = ndl_emb(ndl_idx).to_numpy()

        diff = tch_result - ndl_result
        err=np.linalg.norm(diff)

        #err = np.max(np.abs(diff))
        print(f'{err=}')

        if err < 1e-4:
            print(f"fp32_cuda: {self.GREEN}PASS{self.RESET}")
        else:
            print(f"fp32_cuda: {self.RED}FAILED{self.RESET}, {err=}")

if __name__ == '__main__':
    unittest.main()

