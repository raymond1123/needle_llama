import numpy as np
import needle as ndl

class Tensor:
    def __init__(self, shape, dtype=ndl.fp32, backend=ndl.cpu):
        if dtype == ndl.fp32:
            self.tensor = ndl.TensorFloat(shape, dtype, backend)
        elif dtype == ndl.fp16:
            self.tensor = ndl.TensorHalf(shape, dtype, backend)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.shape = tuple(self.tensor.shape())

    def to_numpy(self):
        return self.tensor.to_numpy()

    def __getattr__(self, name):
        return getattr(self.tensor, name)


