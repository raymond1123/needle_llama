#include <cuda_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "needle_tensor.cuh"
#include "needle_util.cuh"
#include "nn/function.cuh"
#include "nn/nn_module.cuh"
#include "nn/linear.cuh"
#include "nn/nn_embedding.cuh"
#include "nn/module_list.cuh"
#include "init/init_basic.cuh"
#include "init/initial.hpp"


void bind_tensor(py::module& m) {
    py::class_<NdlTensor> tensor_class(m, "Tensor");
    tensor_class
        .def(py::init<py::array_t<float>&, DataType, BackendType>(),
            py::arg("np_array"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("backend")=BackendType::CUDA)
        
        .def_readonly("dtype", &NdlTensor::dtype) // 将 dtype 作为属性公开
        .def_readonly("device", &NdlTensor::device) // 将 device 作为属性公开

        .def("to_numpy", &NdlTensor::to_numpy)
        .def("half", &NdlTensor::half)
        .def("to_float", &NdlTensor::to_float)
        .def("shape", &NdlTensor::shape)
        .def("strides", &NdlTensor::strides)
        .def("matmul", &NdlTensor::matmul)
        .def("rms_norm", &NdlTensor::rms_norm)
        .def("__getitem__", &NdlTensor::slice)
        .def("__setitem__", &NdlTensor::setitem)
        .def("rotary_emb", &NdlTensor::rotary_emb)
        .def("summation", py::overload_cast<const std::vector<int>&>(&NdlTensor::summation),
                        py::arg("axes"))
        .def("summation", py::overload_cast<>(&NdlTensor::summation))
        .def("softmax", py::overload_cast<>(&NdlTensor::softmax))
        .def("argmax", &NdlTensor::argmax,
            py::arg("dim"),
            py::arg("keepdim") = false) 
        .def("transpose", &NdlTensor::transpose)
        .def("reshape", &NdlTensor::reshape)
        .def("contiguous", &NdlTensor::contiguous)
        .def("silu", &NdlTensor::silu)

        .def("__add__", [](NdlTensor& a, NdlTensor& b) {
            return a + b;
        }, py::is_operator())

        .def("__add__", [](NdlTensor& a, float scalar) {
            return a + scalar;
        }, py::is_operator())

        .def("__mul__", [](NdlTensor& a, NdlTensor& b) {
            return a * b;
        }, py::is_operator())

        .def("__mul__", [](NdlTensor& a, float scalar) {
            return a * scalar;
        }, py::is_operator())

        .def("__truediv__", [](NdlTensor& a, NdlTensor& b) {
            return a / b;
        }, py::is_operator())

        .def("__truediv__", [](NdlTensor& a, float scalar) {
            return a / scalar;
        }, py::is_operator())

        // 反向的 operator+，处理 float + NdlTensor
        .def("__radd__", [](NdlTensor& a, float scalar) {
            return a + scalar;
        }, py::is_operator())

        // 使用运算符重载 @ 运算符
        .def("__matmul__", &NdlTensor::matmul)
        .def("from_buffer", &NdlTensor::from_buffer)

        .def_property_readonly("shape", [](NdlTensor& self) {
            const auto& shape_vector = self.shape();  // 获取形状的 std::vector<int32_t>
            py::tuple shape_tuple(shape_vector.size());  // 创建一个与 vector 大小相同的 py::tuple
            for (size_t i = 0; i < shape_vector.size(); ++i) {
                shape_tuple[i] = shape_vector[i];  // 将 vector 的元素拷贝到 tuple 中
            }
            return shape_tuple;
        })

        .def_property_readonly("strides", [](NdlTensor& self) {
            const auto& strides_vector = self.strides();  // 获取形状的 std::vector<int32_t>
            py::tuple strides_tuple(strides_vector.size());  // 创建一个与 vector 大小相同的 py::tuple
            for (size_t i = 0; i < strides_vector.size(); ++i) {
                strides_tuple[i] = strides_vector[i];  // 将 vector 的元素拷贝到 tuple 中
            }
            return strides_tuple;
        })
        ;
}

void bind_module(py::module& m) {
    py::enum_<ModuleOp>(m, "ModuleOp")
        .value("TRAIN", ModuleOp::TRAIN)
        .value("VAL", ModuleOp::VAL)
        .value("TO_HALF", ModuleOp::TO_HALF)
        .export_values();

    py::module nn = m.def_submodule("nn", "Neural network operations");

    class PyModule : public Module {
    public:
        using Module::Module; // 继承构造函数
    
        // 这个函数是用来覆盖纯虚函数 forward 的
        NdlTensor forward(const NdlTensor& tensor) override {
            PYBIND11_OVERLOAD_PURE(
                NdlTensor,       // 返回类型
                Module,          // 父类类型
                forward,         // 函数名（在 C++ 中）
                tensor           // 参数
            );
        }
    };

    py::class_<Module, PyModule, std::shared_ptr<Module>>(nn, "Module")
       .def(py::init<>())
       .def(py::init<std::vector<std::shared_ptr<Module>>, std::string, DataType, BackendType>(),
            py::arg("modules"),
            py::arg("name")="Module",
            py::arg("dtype")=DataType::NOT_CONCERN,
            py::arg("device")=BackendType::NOT_CONCERN)
       .def("train", &Module::train)
       .def("eval", &Module::eval)
       .def("half", &Module::half)
       .def("__call__", &Module::operator())
       ;

    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(nn, "Sequential")
        .def(py::init<std::vector<std::shared_ptr<Module>>&>(), py::arg("modules")) 
        .def("forward", &Sequential::forward); 

    py::class_<Linear, Module, std::shared_ptr<Linear>>(nn, "Linear")
        .def(py::init<int, int, bool, DataType, BackendType, std::string>(),
            py::arg("in_features"),
            py::arg("out_features"),
            py::arg("bias") = true,
            py::arg("dtype") = DataType::FLOAT,
            py::arg("device") = BackendType::CUDA,
            py::arg("name") = "Linear")
        .def("forward", &Linear::forward)
        .def("set_params", &Linear::set_params,
            py::arg("params"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("device")=BackendType::CUDA)
        .def("see_weight", &Linear::see_weight)
        .def_readwrite("weight", &Linear::weight) // 将 dtype 作为属性公开
        .def_readwrite("bias", &Linear::bias) // 将 dtype 作为属性公开
        ;

    py::class_<Embedding, Module, std::shared_ptr<Embedding>>(nn, "Embedding")
        .def(py::init<int32_t, int32_t, DataType, BackendType, std::string>(),
            py::arg("vocab_size"),
            py::arg("dim"),
            py::arg("dtype") = DataType::FLOAT,
            py::arg("device") = BackendType::CUDA,
            py::arg("name") = "Embedding")
        .def("forward", &Embedding::forward)
        .def("to_half", &Embedding::to_half)
        .def("see_weight", &Embedding::see_weight)
        .def("set_params", &Embedding::set_params,
            py::arg("params"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("device")=BackendType::CUDA)
        .def_readwrite("bias", &Embedding::token_emb)
        ;

    // 绑定 ModuleList 类
    py::class_<ModuleList>(nn, "ModuleList")
        .def(py::init<>())
        .def("append", &ModuleList::append)
        .def("forward", &ModuleList::forward)
        .def("__getitem__", &ModuleList::operator[])
        .def("__len__", &ModuleList::size)
        .def("__iter__", [](ModuleList &self) {
            return py::make_iterator(self.begin(), self.end());
        }, py::keep_alive<0, 1>());  // 保持容器有效，直到迭代器耗尽


}

void bind_function(py::module& m) {
    m.def("arange", [](int32_t start, int32_t end, int32_t step = 1, 
                       DataType dtype = DataType::FLOAT, 
                       BackendType device = BackendType::CUDA) {
        return NdlTensor::arange(start, end, step, dtype, device);
    }, 
    py::arg("start"), 
    py::arg("end"), 
    py::arg("step") = 1, 
    py::arg("dtype") = DataType::FLOAT, 
    py::arg("device") = BackendType::CUDA);

    m.def("ones", [](std::vector<int32_t> shape,
                       DataType dtype = DataType::FLOAT, 
                       BackendType device = BackendType::CUDA) {
        return NdlTensor::ones(shape, dtype, device);
    }, 
    py::arg("shape"),
    py::arg("dtype") = DataType::FLOAT, 
    py::arg("device") = BackendType::CUDA);

    m.def("zeros", [](std::vector<int32_t> shape,
                       DataType dtype = DataType::FLOAT, 
                       BackendType device = BackendType::CUDA) {
        return NdlTensor::zeros(shape, dtype, device);
    }, 
    py::arg("shape"),
    py::arg("dtype") = DataType::FLOAT, 
    py::arg("device") = BackendType::CUDA);

    m.def("fill_val", [](std::vector<int32_t> shape,
                        float val,
                        DataType dtype = DataType::FLOAT, 
                        BackendType device = BackendType::CUDA) {
        return NdlTensor::fill_val(shape, val, dtype, device);
    }, 
    py::arg("shape"),
    py::arg("val"),
    py::arg("dtype") = DataType::FLOAT, 
    py::arg("device") = BackendType::CUDA);

    m.def("where", [](const NdlTensor& condition,
                      const NdlTensor& x,
                      const NdlTensor& y) {
        return NdlTensor::where(condition, x, y);
    }, 
    py::arg("condition"),
    py::arg("x"),
    py::arg("y"));
}

PYBIND11_MODULE(needle, m) {
    py::enum_<DataType>(m, "DataType")
        .value("FLOAT", DataType::FLOAT)
        .value("HALF", DataType::HALF)
        .value("NOT_CONCERN", DataType::NOT_CONCERN)
        .export_values();

    // Binding for operator+ with another Tensor
    py::enum_<BackendType>(m, "BackendType")
        .value("CPU", BackendType::CPU)
        .value("CUDA", BackendType::CUDA)
        .export_values();

    m.attr("fp32") = DataType::FLOAT;
    m.attr("fp16") = DataType::HALF;
    m.attr("cuda") = BackendType::CUDA;
    m.attr("cpu") = BackendType::CPU;

    bind_tensor(m);
    bind_function(m);
    bind_module(m);

}


