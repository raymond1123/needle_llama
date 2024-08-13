import numpy as np
import tensor as ts

def test_arange(start, end, step, seq_len):

    npx = np.arange(start, end, step)
    tsx = ts.arange(start, end, step)

    npm = np.arange(seq_len)
    tsm = ts.arange(0, seq_len)

    return tsx, npx, npm, tsm


if __name__ == "__main__":

    start = 0
    end = 64
    step = 2
    seq_len = 2048

    tsx, npx, npm, tsm = test_arange(start, end, step, seq_len)

    print(f'err={np.linalg.norm(tsx.to_numpy()-npx)}')
    print(f'err={np.linalg.norm(tsm.to_numpy()-npm)}')



