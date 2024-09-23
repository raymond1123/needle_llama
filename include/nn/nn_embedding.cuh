#pragma once

#include "common.hpp"
#include "nn/nn_module.cuh"
#include "init/initial.hpp"
#include <cublas_v2.h>

namespace py = pybind11;

class Embedding: public Module {

public:
    using param_type = std::shared_ptr<NdlTensor>;
    using module_type = std::shared_ptr<Module>;

    Embedding(int32_t vocab_size, int32_t dim, 
           DataType dtype=DataType::FLOAT,
           BackendType device=BackendType::CUDA, 
           std::string name="Embedding"): 
        Module(std::vector<module_type>(), name, dtype, device), 
        _vocab_size(vocab_size), _dim(dim) {

        std::vector<int32_t> weight_shape = {_vocab_size, _dim};
        token_emb = kaiming_uniform(weight_shape[0], weight_shape, dtype, device, "relu");
        this->_params.push_back(token_emb);
    }

    void set_params(py::array_t<float>& param,
                    DataType dtype=DataType::FLOAT,
                    BackendType device=BackendType::CUDA) {

        token_emb.reset(new NdlTensor(param, dtype, device));
        if(dtype==DataType::HALF)
            to_half();

        this->dtype = dtype;
        this->_params[0] = token_emb;
    }

    NdlTensor forward(const NdlTensor& tensor) final {
        return token_emb->embedding(tensor);
    }

    void to_half() override {
        if(this->dtype==DataType::FLOAT) {
            this->dtype = DataType::HALF;
            token_emb = std::make_shared<NdlTensor>(std::move(token_emb->half()));
        }

        this->_params[0] = token_emb;
    }

    inline py::array_t<float> see_weight() {
        return token_emb->to_numpy();
    }

public:
    param_type token_emb; // shape=(_in_features, _out_features)

private:
    int32_t _vocab_size;
    int32_t _dim;
};
