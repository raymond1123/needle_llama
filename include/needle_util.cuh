#ifndef __NEEDLE_UTIL_HPP__
#define __NEEDLE_UTIL_HPP__

#include "common.hpp"
#include "ops/generic_op.cuh"
#include "ops/ops_math.hpp"
#include "backend/base_tensor.hpp"

template<typename Dtype>
Tensor<Dtype> stack(std::vector<Tensor<Dtype>>& inputs, int dim=0) {

    std::shared_ptr<GenericOp<Dtype>> stack_op = 
        std::make_shared<StackOp<Dtype>>(dim, OpType::Stack);

    BackendType backend = inputs[0].device();

    std::vector<std::shared_ptr<BaseTensor<Dtype>>> cached_inputs;
    for(auto& input: inputs)
        cached_inputs.push_back(input.cached_data());

    Tensor<Dtype> tensor = Tensor<Dtype>(backend, stack_op, cached_inputs);
    tensor.realized_cached_data(stack_op, cached_inputs);

    return tensor;
}

template<typename Dtype>
std::vector<Tensor<Dtype>> split(Tensor<Dtype>& input, int dim=0) {
    BackendType backend = input.device();

    int split_size = input.shape()[dim];
    std::vector<Tensor<Dtype>> tensors;
    auto cached_input = input.cached_data();

    for(int i=0; i<split_size; ++i) {

        /* different split op */
        std::shared_ptr<GenericOp<Dtype>> split_op =
            std::make_shared<SplitOp<Dtype>>(dim, i, OpType::Split);

        tensors.push_back(Tensor<Dtype>(backend, split_op, {cached_input}));

        std::vector<std::shared_ptr<BaseTensor<Dtype>>> split_input = {cached_input};
        tensors[i].realized_cached_data(split_op, split_input);
    }

    return tensors;
}

#endif

