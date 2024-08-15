#ifndef __FUNCTIONAL_CUH__
#define __FUNCTIONAL_CUH__

#include "tensor.cuh"
#include "needle_tensor.cuh"
#include "ops/bp/padding.cuh"

template<typename Dtype>
using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

template<typename Dtype>
Tensor<Dtype> pad(Tensor<Dtype>& tensor, std::vector<int32_t> axes) {
    return tensor.padding(axes);
}

/*
NdlTensor arange(int start, int end, int step=1, 
                     DataType dtype=DataType::FLOAT, 
                     BackendType device=BackendType::CUDA ) { 

    return NdlTensor::arange(start, end, step, dtype, device);
}
*/

#endif
