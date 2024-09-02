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

        if(_need_bias)
            this->_params[1] = bias;
    }

    NdlTensor forward(const NdlTensor& tensor) final {

        //auto x = tensor;
        int in_feat_idx = tensor.shape().size()- 1;

        /* debug */
        for(int i=0; i<tensor.shape().size(); ++i) {
            printf("tensor.shape[%d]=%d,", i, tensor.shape()[i]);
        }
        printf("\n");

        for(int i=0; i<weight->shape().size(); ++i) {
            printf("weight.shape[%d]=%d,", i, weight->shape()[i]);
        }
        printf("\n");
        /* debug done */

        assert(tensor.shape()[in_feat_idx]==_in_features &&"shape of input tensor and weight does not match");

        const auto& w = *weight;
        auto out = tensor.matmul(w);

        if(_need_bias) {
            out += bias->broadcast_to(out.shape());
        }

        return {out};
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
    int _in_features;
    int _out_features;
};
