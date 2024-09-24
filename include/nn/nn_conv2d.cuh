#pragma once

#include "common.hpp"
#include "nn/nn_module.cuh"
#include "init/initial.hpp"
#include <cublas_v2.h>

namespace py = pybind11;

class Conv2d: public Module {

public:
    using param_type = std::shared_ptr<NdlTensor>;
    using module_type = std::shared_ptr<Module>;

    Conv2d(int in_channels, int out_channels, 
           int kernel_size, int stride=1,
           bool need_bias=true, 
           DataType dtype=DataType::FLOAT,
           BackendType device=BackendType::CUDA, 
           std::string name="Conv2d"): 
        Module(std::vector<module_type>(), name, dtype, device), 
        _in_channels(in_channels), _out_channels(out_channels), 
        _padding((kernel_size-1)/2), _kernel_size(kernel_size), 
        _stride(stride), _need_bias(need_bias) {

            /* weight.shape = (in_channels, out_channels)*/
            std::vector<int32_t> weight_shape = {_kernel_size, _kernel_size, 
                                                 _in_channels, _out_channels};
            weight = kaiming_uniform(_in_channels*_kernel_size*_kernel_size, 
                                     weight_shape, dtype, device, "relu");

            if(_need_bias) {
                float bias_bound = 1/pow((_in_channels*pow(_kernel_size,2.0)), 0.5);
                /* bias.shape = (1, out_channels)*/
                std::vector<int32_t> bias_shape = {1, _out_channels};
                bias = rand_shptr(bias_shape, -bias_bound, bias_bound, dtype, device);
            }

            this->_params.push_back(weight);
            this->_params.push_back(bias);
    }

    ~Conv2d() { 
        for(auto& p: this->_params)
            p.reset();
    }

    void set_params(std::vector<py::array_t<float>>& params,
                    DataType dtype=DataType::FLOAT,
                    BackendType device=BackendType::CUDA) {
        if(_need_bias) {
            assert(params.size()==2 && "param number of Conv2d with bias must be 2");
            bias.reset(new NdlTensor(params[1], dtype, device));
        } else 
            assert(params.size()==1 && "param number of Conv2d without bias must be 1");

        weight.reset(new NdlTensor(params[0], dtype, device));

        _kernel_size = weight->shape()[0];
        _in_channels = weight->shape()[2];
        _out_channels = weight->shape()[3];
        this->dtype = dtype;

        this->_params[0] = weight;

        if(_need_bias)
            this->_params[1] = bias;
    }

    NdlTensor forward(const NdlTensor& tensor) final {

        //const auto x = tensor.permute({0,2,3,1});
        //auto out = x.conv2d(*weight, _stride, _padding);
        NdlTensor out = tensor.conv2d(*weight, _stride, _padding);

        if(_need_bias) {
            std::vector<int32_t> reshape_shape = {1,1,1, _out_channels};
            NdlTensor ndl_bias = (bias->reshape(reshape_shape)).broadcast_to(out.shape());
            out += ndl_bias;
        }

        return out;
    }

    void to_half() override {
        if(this->dtype==DataType::FLOAT) {
            this->dtype = DataType::HALF;
            weight = std::make_shared<NdlTensor>(std::move(weight->half()));

            if(_need_bias)
                bias = std::make_shared<NdlTensor>(std::move(bias->half()));
        }

        this->_params[0] = weight;

        if(_need_bias)
            this->_params[1] = bias;
    }

    inline py::array_t<float> see_weight() {
        return weight->to_numpy();
    }

public:
    param_type weight; // shape=(_in_features, _out_features)
    param_type bias; // shape = (_out_features, 1)

private:
    bool _need_bias;
    int _in_channels;
    int _out_channels;
    int _stride;
    int _padding;
    int _kernel_size;
};
