import numpy as np
import tensor as ts

def test_transpose(shape, trans_np_axes, trans_ts_axes):

    np_x = np.random.randn(*shape)
    ts_x = ts.Tensor(np_array=np_x, backend=ts.BackendType.CUDA)

    trans_npx = np_x.transpose(trans_np_axes)

    trans_tsx = ts_x.transpose(trans_ts_axes)
    trans_tsx.contiguous()

    return trans_tsx, trans_npx


if __name__ == "__main__":

    shape = (3, 2, 4, 3)
    trans_ts_axes = (-1,-2)
    trans_np_axes = (0, 1, -1,-2)

    trans_tsx, trans_npx = test_transpose(shape, trans_np_axes, trans_ts_axes)

    print(f'err={np.linalg.norm(trans_tsx.to_numpy()-trans_npx)}')



