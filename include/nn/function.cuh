#ifndef __FUNCTIONAL_CUH__
#define __FUNCTIONAL_CUH__

#include "tensor.cuh"

#include "ops/bp/padding.cuh"

template<typename Dtype>
using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

template<typename Dtype>
Tensor<Dtype> pad(Tensor<Dtype>& tensor, std::vector<int32_t> axes) {
    return tensor.padding(axes);
}

#endif

