#ifndef __LINEAR_CUH__
#define __LINEAR_CUH__

#include "common.hpp"
#include "nn/nn_module.cuh"
#include "init/initial.hpp"
#include <cublas_v2.h>

namespace py = pybind11;

class Linear: public Module {

public:
    using param_type = std::shared_ptr<NdlTensor>;

    Linear(int in_features, int out_features, 
           bool bias=true, 
           DataType dtype=DataType::FLOAT,
           BackendType device=BackendType::CUDA): 
        Module(), _need_bias(bias),
        _in_features(in_features), _out_features(out_features) {

            std::vector<int32_t> weight_shape = {_out_features, _in_features};
            _weight = kaiming_uniform(weight_shape, dtype, device, "relu");

            if(bias) {
                std::vector<int32_t> bias_shape = {1, _out_features};
                _bias = kaiming_uniform(bias_shape, dtype, device, "relu");
            }

            this->_params.push_back(_weight);
            this->_params.push_back(_bias);
    }

    void set_params(std::vector<py::array_t<float>>& params,
                    DataType dtype,
                    BackendType device=BackendType::CUDA) {
        if(_need_bias) {
            assert(params.size()==2 && "param number of Linear with bias must be 2");
            _bias.reset(new NdlTensor(params[1], dtype, device));
        } else 
            assert(params.size()==1 && "param number of Linear without bias must be 1");

        _weight.reset(new NdlTensor(params[0], dtype, device));
        _in_features = _weight->shape()[1];
        _out_features = _weight->shape()[0];

        this->_params.push_back(_weight);
        this->_params.push_back(_bias);
    }

    virtual std::vector<NdlTensor> forward(std::vector<NdlTensor>& tensors) override {
        assert(tensors.size()==1 && "input number of Linear must be 1");

        auto x = tensors[0];
        int in_feat_idx = x.shape().size()- 1;
        assert(x.shape()[in_feat_idx]==_in_features &&"shape of input tensor and weight does not match");

        /* y = x@A.T + b */
        auto weight_T = _weight->transpose({-1, -2});
        auto out = x.matmul(weight_T);

        if(_need_bias) {
            out += _bias->broadcast_to(out.shape());
        }

        return {out};
    }

private:
    param_type _weight; // shape=(_in_features, _out_features)
    param_type _bias; // shape = (_out_features, 1)

    bool _need_bias;
    int _in_features;
    int _out_features;
};

#endif

