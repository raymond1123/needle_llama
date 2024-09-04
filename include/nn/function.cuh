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

template<typename Dtype>
Tensor<Dtype> f_where(const Tensor<float>& condition,
                    const Tensor<Dtype>& x, 
                    const Tensor<Dtype>& y) {

    assert((condition.shape()==x.shape() && x.shape()==y.shape()) &&
           "shape of condition, x and y in where op must equal");

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<WhereOp<Dtype>>(condition.cached_data(),
                                         OpType::Where);

    std::vector<cached_data_type<Dtype>> inputs;
    inputs.push_back(x.cached_data());
    inputs.push_back(y.cached_data());

    return (*op)(op, inputs, y.device);
}

/*
NdlTensor arange(int start, int end, int step=1, 
                     DataType dtype=DataType::FLOAT, 
                     BackendType device=BackendType::CUDA ) { 

    return NdlTensor::arange(start, end, step, dtype, device);
}
*/

#endif
