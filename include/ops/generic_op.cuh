#ifndef __GENERIC_OP__
#define __GENERIC_OP__

#include "common.hpp"
#include "ops/ops_util.cuh"

template<typename Dtype> class Tensor;
template<typename Dtype> class BaseTensor;

template<typename Dtype>
class GenericOp {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    GenericOp(OpType op_type):_op_type(op_type) {};

    virtual cached_data_type compute(std::vector<cached_data_type> inputs)=0;

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor)=0;

    inline Tensor<Dtype> operator()(const std::shared_ptr<GenericOp<Dtype>> op,
                                    std::vector<cached_data_type>& inputs,
                                    BackendType backend) const {
        return Tensor<Dtype>::make_from_op(op, inputs, backend);
    }

    inline cached_data_type operator()(const std::shared_ptr<GenericOp<Dtype>> op,
                                    std::vector<cached_data_type>& inputs,
                                    BackendType backend, bool op_on_self) const {
        return Tensor<Dtype>::make_from_op_on_self(op, inputs, backend, op_on_self);
    }

    inline int op_type() {return static_cast<int>(_op_type);}

protected:
    virtual inline cudaError_t _get_num_blocks()=0;
public:
    OpType _op_type;
};

#endif

///* input two tensor and two int; output one Tensor */
//23. template<Dtype> Tensor<Dtype> conv(const Tensor<Dtype>& a, 
//                                  const Tensor<Dtype>& b, 
//                                  const int stride=1, 
//                                  const int padding=1);

