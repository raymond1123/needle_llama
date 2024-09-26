#ifndef __INITIAL_CUH__
#define __INITIAL_CUH__

#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

#include "needle_tensor.cuh"
#include "init/init_basic.cuh"

namespace py = pybind11;

std::shared_ptr<NdlTensor> xavier_uniform(std::vector<int32_t> shape, 
                             float gain=1.0,
                             DataType dtype=DataType::FLOAT,
                             BackendType device=BackendType::CUDA, 
                             std::optional<int> seed = std::nullopt) {

    assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");

    auto scope = [](int32_t fan_in, int32_t fan_out) { 
        float root = 6.0/(float(fan_in)+float(fan_out));
        return sqrt(root);
    };

    float high = gain*scope(shape[0], shape[1]);

    return rand_shptr(shape, seed, -high, high, dtype, device);
}

std::shared_ptr<NdlTensor> xavier_normal(std::vector<int32_t> shape, 
                             float gain=1.0, 
                             DataType dtype=DataType::FLOAT,
                             BackendType device=BackendType::CUDA,
                             std::optional<int> seed = std::nullopt) {

    assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");

    auto std_f = [](int32_t fan_in, int32_t fan_out) { 
        float root = 2.0/(float(fan_in)+float(fan_out));
        return sqrt(root);
    };

    float std = gain*std_f(shape[0], shape[1]);

    return randn_shptr(shape, seed, 0.0, std, dtype, device);
}

std::shared_ptr<NdlTensor> kaiming_uniform(int32_t fan_in,
                              std::vector<int32_t> shape, 
                              DataType dtype=DataType::FLOAT,
                              BackendType device=BackendType::CUDA,
                              std::string nonlinearity="relu",
                              std::optional<int> seed = std::nullopt) {

    //assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");
    assert(nonlinearity=="relu" && "only relu supported currently");

    auto scope = [](int32_t fan_in) { 
        float gain = sqrt(2.0);
        return gain*sqrt(3.0/(float)fan_in);
    };

    float high = scope(fan_in);

    return rand_shptr(shape, seed, -high, high, dtype, device);
}

std::shared_ptr<NdlTensor> kaiming_normal(std::vector<int32_t> shape,
                             DataType dtype=DataType::FLOAT,
                             BackendType device=BackendType::CUDA,
                             std::string nonlinearity="relu",
                             std::optional<int> seed = std::nullopt) {

    assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");
    assert(nonlinearity=="relu" && "only relu supported currently");

    auto std_f = [](int32_t fan_in) { 
        float gain = sqrt(2.0);
        return gain/sqrt((float)fan_in);
    };

    float std = std_f(shape[0]);

    return randn_shptr(shape, seed, 0.0, std, dtype, device);
}

#endif

