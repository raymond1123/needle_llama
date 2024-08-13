import numpy as np
import tensor as ts
import torch

if __name__ == "__main__":

    shape = (2,1,3)
    np.random.seed(100202)

    arr = np.random.randn(*shape)
    brr = np.random.randn(*shape)
    #arr = np.arange(6).reshape(shape)
    #brr = arr+3.1234324235

    ft1 = ts.Tensor(np_array=arr, backend=ts.BackendType.CUDA)
    ft2 = ts.Tensor(np_array=brr, backend=ts.BackendType.CUDA)

    tt1 = torch.from_numpy(arr).to(torch.float32).requires_grad_(True)
    tt2 = torch.from_numpy(brr).to(torch.float32).requires_grad_(True)


    ft3 = ft1 + ft2
    tt3 = tt1 + tt2

    ft4 = ft3 - ft1
    tt4 = tt3 - tt1

    ft5 = ft4 * ft3
    tt5 = tt4 * tt3

    ft6 = ft5+ft3
    tt6 = tt5+tt3

    ft6 += ft5
    tt6 += tt5

    ft7 = ft6.reshape((2,3,1))
    tt7 = tt6.reshape((2,3,1))

    ft8 = ft6.reshape((2,3,1)) + ft7 # shape = (2,3,1)
    tt8 = tt6.reshape((2,3,1)) + tt7

    bshape = (4,2,3,2) # broadcast_axes=(0,3)
    ft9 = ft8.broadcast_to(bshape) # shape=(4,2,3,2)
    tt9 = tt8.broadcast_to(bshape)

    axes = (3,0,2,1)
    ft10 = ft9.permute(axes)
    tt10 = tt9.permute(axes)

    t_axes = (0,2)
    ft11 = ft10.transpose(t_axes)
    tt11 = tt10.transpose(*t_axes)

    ft12 = ft11.sum()
    tt12 = tt11.sum()

    ft_list = [ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12]
    tt_list = [tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tt10, tt11, tt12]

    for i in range(len(ft_list)):
        print(f'ft{i+1}, \
              eeeeeeer={np.linalg.norm(ft_list[i].to_numpy() - tt_list[i].detach().numpy())}')

    print('bbbbbbbbbbbbbbb\n\n')
    ft12.backward()
    tt12.backward()

    print(f'eeeeeeer={np.linalg.norm(ft1.grad() - tt1.grad.numpy())}')
    print(f'eeeeeeer={np.linalg.norm(ft2.grad() - tt2.grad.numpy())}')

    print()


