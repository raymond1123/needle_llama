#ifndef __INIT_BASIC_CUH__
#define __INIT_BASIC_CUH__

#include "tensor.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;

// Generate uniformly distributed random numbers
py::array_t<float> _generate_uniform(std::vector<int32_t>& shape, 
                                    float min=0.0, float max=1.0) {
    size_t size=1;
    for(auto& s: shape)
        size *= s;

    // Initialize a random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);

    // Create a buffer to hold the random numbers
    auto result = py::array_t<float>(size);

    auto ptr = result.mutable_data();
    for (int i = 0; i < size; ++i) {
        ptr[i] = dist(rng);
    }

    result.resize(shape);
    return result;
}

py::array_t<float> _generate_normal(std::vector<int32_t> shape, 
                   float mean=0.0, float std=1.0) {
    size_t size=1;
    for(auto& s: shape)
        size *= s;

    // Initialize a random number generator
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(mean, std);

    // Create a buffer to hold the random numbers
    auto result = py::array_t<float>(size);

    auto ptr = result.mutable_data();
    for (int i = 0; i < size; ++i) {
        ptr[i] = dist(rng);
    }

    result.resize(shape);
    return result;
}

py::array_t<float> _generate_randb(std::vector<int32_t>& shape, 
                                   float prob=0.5, float min=0.0, float max=1.0) {
    size_t size=1;
    for(auto& s: shape)
        size *= s;

    // Initialize a random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);

    // Create a buffer to hold the random numbers
    auto result = py::array_t<float>(size);

    auto ptr = result.mutable_data();
    for (int i = 0; i < size; ++i) {
        if(dist(rng)>=prob) ptr[i] = 1.0;
        else ptr[i] = 0.0;
    }

    result.resize(shape);
    return result;
}


// Generate uniformly distributed random numbers
template<typename Dtype>
Tensor<Dtype> rand(std::vector<int32_t> shape, 
                   float min=0.0, float max=1.0,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_uniform(shape, max, min);
    auto tensor = Tensor<Dtype>(result, dtype, device);

    return tensor;
}

// Generate uniformly distributed random numbers
template<typename Dtype>
std::shared_ptr<Tensor<Dtype>> rand_shptr(std::vector<int32_t> shape, 
                   float min=0.0, float max=1.0,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_uniform(shape, max, min);
    std::shared_ptr<Tensor<Dtype>> tensor = 
        std::make_shared<Tensor<Dtype>>(new Tensor<Dtype>(result, dtype, device));

    return tensor;
}

// Generate Gaussian distributed random numbers
template<typename Dtype>
Tensor<Dtype> randn(std::vector<int32_t> shape, 
                   float mean=0.0, float std=1.0,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_normal(shape, mean, std);
    auto tensor = Tensor<Dtype>(result, dtype, device);
    return tensor;
}

template<typename Dtype>
std::shared_ptr<Tensor<Dtype>> randn_shptr(std::vector<int32_t> shape, 
                   float mean=0.0, float std=1.0,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_normal(shape, mean, std);
    std::shared_ptr<Tensor<Dtype>> tensor = 
        std::make_shared<Tensor<Dtype>>(new Tensor<Dtype>(result, dtype, device));

    return tensor;
}

template<typename Dtype>
Tensor<Dtype> randb(std::vector<int32_t> shape, 
                    float prob=0.5,
                    DataType dtype=DataType::FLOAT,
                    BackendType device=BackendType::CUDA) {

    auto result = _generate_randb(shape, prob);
    auto tensor = Tensor<Dtype>(result, dtype, device);
    return tensor;
}

template<typename Dtype>
Tensor<Dtype> ones(std::vector<int32_t> shape, 
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {
    return Tensor<Dtype>::ones(shape, dtype, device);
}

template<typename Dtype>
Tensor<Dtype> ones_like(Tensor<Dtype>& input) {
    return Tensor<Dtype>::ones(input.shape(), input.dtype, input.device());
}

template<typename Dtype>
Tensor<Dtype> zeros(std::vector<int32_t> shape, 
                    DataType dtype=DataType::FLOAT,
                    BackendType device=BackendType::CUDA) {
    return Tensor<Dtype>::zeros(shape, dtype, device);
}

template<typename Dtype>
Tensor<Dtype> arange(size_t start, size_t end, size_t step=1, 
                     DataType dtype=DataType::FLOAT,
                     BackendType device=BackendType::CUDA) {
    return Tensor<Dtype>::arange(start, end, step, dtype, device);
}

template<typename Dtype>
Tensor<Dtype> zeros_like(Tensor<Dtype>& input) {
    return Tensor<Dtype>::zeros(input.shape(), input.dtype, input.device());
}

template<typename Dtype>
Tensor<Dtype> constant(std::vector<int32_t> shape, 
                       Dtype val,
                       DataType dtype=DataType::FLOAT,
                       BackendType device=BackendType::CUDA) {
    return Tensor<Dtype>::fill_val(shape, val, dtype, device);
}

template<typename Dtype>
Tensor<Dtype> one_hot(int32_t size, int idx,
                      DataType dtype=DataType::FLOAT,
                      BackendType device=BackendType::CUDA) {

    idx = idx>=0?idx:size+idx;

    auto p_arr = py::array_t<float>(size);
    auto ptr = p_arr.mutable_data();

    for (int i = 0; i < size; ++i)
        ptr[idx] = static_cast<Dtype>(0.0);

    ptr[idx] = static_cast<Dtype>(1.0);
    std::vector<int32_t> shape = {1,size};
    p_arr.resize(shape);

    auto tensor = Tensor<Dtype>(p_arr, dtype, device);
    return tensor;
}

#endif

