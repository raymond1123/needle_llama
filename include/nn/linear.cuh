#pragma once

#include "common.hpp"
#include "nn/nn_module.cuh"
#include "init/initial.hpp"
#include <cublas_v2.h>

namespace py = pybind11;

class Linear: public Module {

public:
    using param_type = std::shared_ptr<NdlTensor>;
    using module_type = std::shared_ptr<Module>;

    Linear(int in_features, int out_features, 
           bool need_bias=true, 
           DataType dtype=DataType::FLOAT,
           BackendType device=BackendType::CUDA, 
           std::string name="Linear"): 
        Module(std::vector<module_type>(), name, dtype, device), 
        _need_bias(need_bias), 
        _in_features(in_features), _out_features(out_features) {

            /* weight.shape = (out_feat, in_feat)*/
            std::vector<int32_t> weight_shape = {_in_features, _out_features};
            weight = kaiming_uniform(weight_shape, dtype, device, "relu");

            if(need_bias) {
                /* bias.shape = (1, out_feat)*/
                std::vector<int32_t> bias_shape = {1, _out_features};
                bias = kaiming_uniform(bias_shape, dtype, device, "relu");
            }

            this->_params.push_back(weight);
            this->_params.push_back(bias);
    }

    void set_params(std::vector<py::array_t<float>>& params,
                    DataType dtype=DataType::FLOAT,
                    BackendType device=BackendType::CUDA) {
        if(_need_bias) {
            assert(params.size()==2 && "param number of Linear with bias must be 2");
            bias.reset(new NdlTensor(params[1], dtype, device));
        } else 
            assert(params.size()==1 && "param number of Linear without bias must be 1");

        weight.reset(new NdlTensor(params[0], dtype, device));
        _in_features = weight->shape()[0];
        _out_features = weight->shape()[1];
        this->dtype = dtype;

        this->_params[0] = weight;
        this->_params[1] = bias;
    }

    NdlTensor forward(const NdlTensor& tensor) final {

        auto x = tensor;
        int in_feat_idx = x.shape().size()- 1;

        /* debug */
        for(int i=0; i<x.shape().size(); ++i) {
            printf("x.shape[%d]=%d,", i, x.shape()[i]);
        }
        printf("\n");

        for(int i=0; i<weight->shape().size(); ++i) {
            printf("weight.shape[%d]=%d,", i, weight->shape()[i]);
        }
        printf("\n");
        /* debug done */

        assert(x.shape()[in_feat_idx]==_in_features &&"shape of input tensor and weight does not match");

        const auto& w = *weight;
        auto out = x.matmul(w);

        if(_need_bias) {
            out += bias->broadcast_to(out.shape());
        }

        return {out};
    }

    void to_half() override {
        if(this->dtype==DataType::FLOAT) {
            this->dtype = DataType::HALF;
            weight = std::make_shared<NdlTensor>(std::move(weight->half()));
            bias = std::make_shared<NdlTensor>(std::move(bias->half()));
        }

        this->_params[0] = weight;
        this->_params[1] = bias;
    }

    void ttt() override {
        printf("aaaa\n");
    } 

    inline py::array_t<float> weight_to_numpy() {
        auto www = this->_params[0];
        return www->to_numpy();
    }

    inline py::array_t<float> bias_to_numpy() {
        auto bbb = this->_params[1];
        return bbb->to_numpy();
    }

public:
    param_type weight; // shape=(_in_features, _out_features)
    param_type bias; // shape = (_out_features, 1)

private:
    bool _need_bias;
    int _in_features;
    int _out_features;
};
