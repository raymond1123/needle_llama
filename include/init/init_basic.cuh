#ifndef __INIT_BASIC_CUH__
#define __INIT_BASIC_CUH__

#include "needle_tensor.cuh"

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
NdlTensor rand(std::vector<int32_t> shape, 
               float min=0.0, float max=1.0,
               DataType dtype=DataType::FLOAT,
               BackendType device=BackendType::CUDA) {

    auto result = _generate_uniform(shape, max, min);
    auto tensor = NdlTensor(result, dtype, device);

    return tensor;
}

// Generate uniformly distributed random numbers
std::shared_ptr<NdlTensor> rand_shptr(std::vector<int32_t> shape, 
                   float min=0.0, float max=1.0,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_uniform(shape, max, min);
    std::shared_ptr<NdlTensor> tensor = 
        std::make_shared<NdlTensor>(result, dtype, device);

    return tensor;
}

// Generate Gaussian distributed random numbers
NdlTensor randn(std::vector<int32_t> shape, 
                   float mean=0.0, float std=1.0,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_normal(shape, mean, std);
    auto tensor = NdlTensor(result, dtype, device);
    return tensor;
}

std::shared_ptr<NdlTensor> randn_shptr(std::vector<int32_t> shape, 
                   float mean=0.0, float std=1.0,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_normal(shape, mean, std);
    std::shared_ptr<NdlTensor> tensor = 
        std::make_shared<NdlTensor>(result, dtype, device);

    return tensor;
}

NdlTensor randb(std::vector<int32_t> shape, 
                    float prob=0.5,
                    DataType dtype=DataType::FLOAT,
                    BackendType device=BackendType::CUDA) {

    auto result = _generate_randb(shape, prob);
    auto tensor = NdlTensor(result, dtype, device);
    return tensor;
}

NdlTensor ones(std::vector<int32_t> shape, 
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {
    return NdlTensor::ones(shape, dtype, device);
}

NdlTensor ones_like(NdlTensor& input) {
    return NdlTensor::ones(input.shape(), input.dtype, input.device());
}

NdlTensor zeros(std::vector<int32_t> shape, 
                    DataType dtype=DataType::FLOAT,
                    BackendType device=BackendType::CUDA) {
    return NdlTensor::zeros(shape, dtype, device);
}

NdlTensor zeros_like(NdlTensor& input) {
    return NdlTensor::zeros(input.shape(), input.dtype, input.device());
}

NdlTensor constant(std::vector<int32_t> shape, 
                   float val,
                   DataType dtype=DataType::FLOAT,
                   BackendType device=BackendType::CUDA) {
    return NdlTensor::fill_val(shape, val, dtype, device);
}

NdlTensor one_hot(int32_t size, int idx,
                  DataType dtype=DataType::FLOAT,
                  BackendType device=BackendType::CUDA) {

    idx = idx>=0?idx:size+idx;

    auto p_arr = py::array_t<float>(size);
    auto ptr = p_arr.mutable_data();

    for (int i = 0; i < size; ++i)
        ptr[idx] = 0.0; 

    ptr[idx] = 1.0; 
    std::vector<int32_t> shape = {1,size};
    p_arr.resize(shape);

    auto tensor = NdlTensor(p_arr, dtype, device);
    return tensor;
}

#endif

