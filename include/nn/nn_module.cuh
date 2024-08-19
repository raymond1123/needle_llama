#ifndef __NN_MODULE_CUH__
#define __NN_MODULE_CUH__

#include "needle_tensor.cuh"
#include "backend/base_tensor.hpp"

static std::unordered_map<std::string, int> SubModulesMap = {
        {"<class 'needle.nn.Module'>", 0},
        {"<class 'needle.nn.Sequential'>", 1},
        {"<class 'needle.nn.Linear'>", 2}
    };

/* special Tensor that represents parameters */
class Parameter: public NdlTensor {};

class Module {
public:
    using module_type = std::shared_ptr<Module>;
    //using module_type = Module;
    using param_type = std::shared_ptr<NdlTensor>;

    Module():_name("Module"), 
             dtype(DataType::NOT_CONCERN), 
             device(BackendType::NOT_CONCERN) {}

    Module(const std::vector<module_type>& modules, 
           std::string name="Module",
           DataType dtype=DataType::NOT_CONCERN,
           BackendType device=BackendType::NOT_CONCERN):_name(name), dtype(dtype), device(device) {
            
            _get_submodules(modules);
            __check();
    }

    virtual ~Module() = default;

    inline void train() {
        _get_submodules();
        _training = true;
        _children(ModuleOp::TRAIN);
    }

    inline void eval() {
        _get_submodules();
        _training = false;
        _children(ModuleOp::VAL);
    }

    inline void half() {
        _get_submodules();
        to_half();
        _children(ModuleOp::TO_HALF);
    }

    virtual NdlTensor forward(const NdlTensor& tensor)=0;
    inline NdlTensor operator()(NdlTensor& input) {
        return forward(input);
    }

    virtual void ttt() {}; 

    virtual void to_half() {
        if(this->dtype==DataType::HALF || _params.size()==0) return;

        if(this->dtype==DataType::FLOAT) {
            this->dtype = DataType::HALF;
            for(auto& param: this->_params)
                param = std::make_shared<NdlTensor>(std::move(param->half()));
        }
        //printf("%s: to_half", _name.c_str());
    }


protected:
    void _get_submodules() {
        if(!_sub_modules.empty()) return;

        /* get __dict__ attributes */
        py::object self = py::cast(this); 
        if (py::hasattr(self, "__dict__")) {
            py::dict dict = self.attr("__dict__");

            for (auto& item : dict) {
                std::string key = py::str(item.first);
                std::string sub_module_name = std::string(py::str(py::type::of(item.second)));
                if (SubModulesMap.find(sub_module_name) != SubModulesMap.end())
                    _sub_modules.emplace(py::cast<module_type>(item.second), key);
            }
        }
    }

    void _get_submodules(const std::vector<module_type>& modules) {
        for(int i=0; i < modules.size(); ++i)
            _sub_modules.emplace(modules[i], std::to_string(i)+"_sub_module");
    }

    void _children(ModuleOp mo) {
        _child_modules(mo);
    }

    void _child_modules(ModuleOp mo) {
        /* recursive terminated */
        if(_sub_modules.empty()) return;

        for(auto& module: _sub_modules) {
            switch (mo) {
            case ModuleOp::TRAIN:
                module.first->_training = true;
                break;
            case ModuleOp::VAL:
                module.first->_training = false;
                break;
            case ModuleOp::TO_HALF:
                module.first->to_half();
                break;
            default:
                throw std::runtime_error("incorrect module operations");
                break;
            }
        }

        for(auto& s: _sub_modules)
            s.first->_child_modules(mo);
    }


private:
    void __check() {
        if(_sub_modules.empty()) return;

        auto m_dtype = _sub_modules.begin()->first->dtype;
        auto m_device= _sub_modules.begin()->first->device;

        for(auto& module: _sub_modules) {
            assert(module.first->dtype==m_dtype && "module dtype does not match");
            assert(module.first->device==m_device && "module device does not match");
        }
    }

public:
    DataType dtype;
    BackendType device;

protected:
    bool _training;
    //std::vector<module_type> _sub_modules;
    std::unordered_map<module_type, std::string> _sub_modules;
    std::vector<param_type> _params;
    std::string _name;
};

class Sequential: public Module {
public:
    using module_type = std::shared_ptr<Module>;

    Sequential(std::vector<module_type>& modules): Module(modules, "Sequential") {}

    NdlTensor forward(const NdlTensor& tensor) override {
        NdlTensor input = tensor;
        NdlTensor out;

        for(auto& module: this->_sub_modules) {
            out = module.first->forward(input);
            input = out;
        }

        return out;
    }
    void ttt() override {} 
};

#endif

