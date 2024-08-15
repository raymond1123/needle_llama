#ifndef __NN_MODULE_CUH__
#define __NN_MODULE_CUH__

#include "needle_tensor.cuh"
#include "backend/base_tensor.hpp"

/* special Tensor that represents parameters */
class Parameter: public NdlTensor {};

class Module {
public:
    using module_type = std::shared_ptr<Module>;
    using param_type = std::shared_ptr<NdlTensor>;

    Module(): _training(true) {}
    Module(std::vector<module_type>& modules): 
        _training(true), _sub_modules(modules) {}

    inline void train() {
        _training = true;
        _children();

        for(auto& m: get_modules())
            m->_training = true;
    }

    inline void eval() {
        _training = false;
        _children();

        for(auto& m: get_modules())
            m->_training = false;
    }

    inline std::vector<NdlTensor> operator()(std::vector<NdlTensor>& inputs) {
        return forward(inputs);
    }

    std::vector<param_type> parameters() {

        std::vector<param_type> params;

        for(int i = 0; i<get_modules().size(); ++i) {
            for(auto& p: get_modules()[i]->_params)
                _params.push_back(p);
        }

        return params;
    }

    virtual std::vector<NdlTensor> forward(std::vector<NdlTensor>& tensors) {
        std::vector<NdlTensor> inputs = tensors;
        std::vector<NdlTensor> out;

        for(int i = 0; i<get_modules().size(); ++i) {
            out = get_modules()[i]->forward(inputs);
            inputs = out;
        }

        return out;
    };

protected:
    void _children() {
        __child_modules(_sub_modules);
    }

    static std::vector<module_type>& get_modules() {
        static std::vector<module_type> _modules;
        return _modules;
    }

private:
    void __child_modules(std::vector<module_type> modules) {
        if(modules.size()==1 && modules[0]->_sub_modules.size()==0) {
            get_modules().push_back(modules[0]);
            return;
        }

        for(auto& s: _sub_modules)
            s->__child_modules({s});
    }

protected:
    bool _training;
    std::vector<module_type> _sub_modules;
    std::vector<param_type> _params;
};

class Sequential: public Module {
public:
    using module_type = std::shared_ptr<Module>;

    Sequential(std::vector<module_type>& modules): Module(modules) {}

    virtual std::vector<NdlTensor> forward(std::vector<NdlTensor>& tensors) override {
        std::vector<NdlTensor> inputs = tensors;
        std::vector<NdlTensor> out;

        for(int i = 0; i<Module::get_modules().size(); ++i) {
            out = Module::get_modules()[i]->forward(inputs);
            inputs = out;
        }

        return out;
    }
};

#endif

