import numpy as np
import tensor as ts
import tensor.nn as ts_nn
import torch
import torch.nn as nn

class Modle(ts_nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.model = ts_nn.Sequential([ts_nn.Linear(out_feat, in_feat),
                                       ts_nn.Linear(in_feat, out_feat),
                                       ts_nn.Linear(out_feat, 10)])

    def forward(self, x):
        self.model.train()
        y = self.model([x])

        return y[0]

def pytorch_linear():

    shape1 = (2, 64,128)
    shape2 = (128,256)
    shape3 = (256, 10)

    #batch_shape1 = (2,64,128)

    arr = np.random.randn(*shape1)
    ts_x = ts.Tensor(np_array=arr, backend=ts.BackendType.CUDA)
    torch_x = torch.from_numpy(arr).to(torch.float32).requires_grad_(True)

    ts_layers = []
    torch_layers = []

    # Define Linear layer
    torch_layer1 = nn.Linear(*shape2)  # Input size: 100, output size: 50
    ts_layer1 = ts_nn.Linear(*shape2)

    weight1 = torch_layer1.weight.data.numpy()
    bias1 = torch_layer1.bias.data.numpy()
    print(f'{weight1.shape=}')

    ts_layer1.set_params([weight1, bias1])

    ts_layers.append(ts_layer1)
    torch_layers.append(torch_layer1)

    torch_layer2 = nn.Linear(*shape3)  # Input size: 100, output size: 50
    ts_layer2 = ts_nn.Linear(*shape3)

    weight2 = torch_layer2.weight.data.numpy()
    bias2 = torch_layer2.bias.data.numpy()
    ts_layer2.set_params([weight2, bias2])

    ts_layers.append(ts_layer2)
    torch_layers.append(torch_layer2)

    ts_sequence = ts_nn.Sequential(ts_layers)
    torch_sequence = nn.ModuleList(torch_layers)

    torch_f = torch_sequence[0](torch_x)
    torch_f = torch_sequence[1](torch_f)

    ts_sequence.train()
    ts_y = ts_sequence([ts_x])[0]

    torch_y = torch_f.sum()
    ts_y = ts_y.sum()

    print(f'{torch_y=}')
    print(f'{ts_y.to_numpy()=}')

    ts_y.backward()
    torch_y.backward()

    return torch_y, ts_y, torch_x.grad.numpy(), ts_x.grad()


if __name__ == "__main__":
    '''
    shape = (256,128)
    arr = np.random.randn(*shape)
    x = ts.Tensor(np_array=arr, backend=ts.BackendType.CUDA)

    model = Modle(*shape)
    y = model.forward(x)
    print(f'aaaaaa: {y.shape()}')
    #print(f'{y.to_numpy()=}')

    y.backward()

    #z = y.sum()
    #z.backward()

    #print(f'{x.grad()}')
    '''

    torch_y, ts_y, torch_grad, ts_grad = pytorch_linear()

    print(f'forward err={np.linalg.norm(ts_y.to_numpy()-torch_y.detach().numpy())}')
    print(f'leaf gradient err={np.linalg.norm(ts_grad - torch_grad)}')

