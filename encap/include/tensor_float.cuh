#include <cuda_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tensor.cuh"
#include "needle_util.cuh"
#include "nn/function.cuh"
#include "nn/nn_module.cuh"
#include "nn/linear.cuh"
#include "init/init_basic.cuh"
#include "init/initial.hpp"

namespace py = pybind11;

void bind_operator_iplus_tensor(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__iadd__",
        [](Tensor<float>& self, Tensor<float>& other) {
            self += other;
            return self;
        });
}

void bind_operator_plus_tensor(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__add__",
        [](Tensor<float>& self, Tensor<float>& other) {
            return self + other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_plus_scalar(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__add__",
        [](Tensor<float>& self, const float scalar) {
            return self + scalar;
        });
}

void bind_operator_sub_tensor(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__sub__",
        [](Tensor<float>& self, Tensor<float>& other) {
            return self - other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_sub_scalar(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__sub__",
        [](Tensor<float>& self, const float scalar) {
            return self - scalar;
        });
}

void bind_operator_mul_tensor(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__mul__",
        [](Tensor<float>& self, Tensor<float>& other) {
            return self * other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_mul_scalar(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__mul__",
        [](Tensor<float>& self, const float scalar) {
            return self * scalar;
        });
}

void bind_operator_div_tensor(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__truediv__",
        [](Tensor<float>& self, Tensor<float>& other) {
            return self / other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_div_scalar(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__truediv__",
        [](Tensor<float>& self, const float scalar) {
            return self / scalar;
        });
}

void bind_operator_pow_tensor(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__pow__",
        [](Tensor<float>& self, Tensor<float>& other) {
            return self.op_pow(other);
        });
}

void bind_operator_pow_scalar(py::class_<Tensor<float>>& tensor_class) {
    tensor_class.def("__pow__",
        [](Tensor<float>& self, const float scalar) {
            return self.op_pow(scalar);
        });
}

void bind_tensor_float(py::module &m, const char *name) {

    py::class_<Tensor<float>> tensor_class(m, name);
    tensor_class
        .def(py::init<py::array_t<float>&, DataType, BackendType>(),
            py::arg("np_array"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("backend")=BackendType::CUDA)

        .def(py::init<std::vector<int32_t>, DataType, BackendType>(),
            py::arg("shape"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("backend")=BackendType::CUDA)

        .def("reshape", &Tensor<float>::reshape)
        .def("flip", &Tensor<float>::flip)
        .def("__getitem__", &Tensor<float>::slice)
        .def("__setitem__", &Tensor<float>::setitem)
        .def("broadcast_to", &Tensor<float>::broadcast_to)
        .def("permute", &Tensor<float>::permute)
        .def("transpose", &Tensor<float>::transpose)
        .def("sum", (Tensor<float> (Tensor<float>::*)(std::vector<int>)) &Tensor<float>::summation, "Summation with specified axes")
        .def("sum", (Tensor<float> (Tensor<float>::*)()) &Tensor<float>::summation, "Summation without specified axes")
        .def("max", &Tensor<float>::max, py::arg("dim"), py::arg("keepdim")=false)
        .def("relu", &Tensor<float>::relu)
        .def("tanh", &Tensor<float>::tanh)
        .def("log", &Tensor<float>::log)
        .def("exp", &Tensor<float>::exp)
        .def("neg", &Tensor<float>::neg)
        .def("__matmul__", &Tensor<float>::matmul)
        .def("dilate", &Tensor<float>::dilate, py::arg("dilation"), py::arg("axes"))
        .def("to_numpy", &Tensor<float>::to_numpy)
        .def("device", &Tensor<float>::device)
        .def("shape", &Tensor<float>::shape)
        .def("size", &Tensor<float>::size)
        .def("strides", &Tensor<float>::strides)
        .def("offset", &Tensor<float>::offset)
        .def("from_buffer", &Tensor<float>::from_buffer)
        .def("contiguous", &Tensor<float>::contiguous)
        .def("backward", &Tensor<float>::backward)
        .def("grad", &Tensor<float>::grad)
        ;

    /* operators */
    bind_operator_plus_tensor(tensor_class);
    bind_operator_plus_scalar(tensor_class);

    bind_operator_sub_tensor(tensor_class);
    bind_operator_sub_scalar(tensor_class);

    bind_operator_mul_tensor(tensor_class);
    bind_operator_mul_scalar(tensor_class);

    bind_operator_div_tensor(tensor_class);
    bind_operator_div_scalar(tensor_class);

    bind_operator_pow_tensor(tensor_class);
    bind_operator_pow_scalar(tensor_class);

    bind_operator_iplus_tensor(tensor_class);
}

void bind_init_float(py::module &m) {
    m.def("rand", &rand<float>, 
          py::arg("shape"), py::arg("min")=0.0, 
          py::arg("max")=1.0, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA,
          "Generate uniformly distributed random tensor");

    m.def("randn", &randn<float>, 
          py::arg("shape"), py::arg("mean")=0.0, 
          py::arg("std")=1.0, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA, 
          "Generate Gaussian distributed random tensor");

    m.def("randb", &randb<float>, 
          py::arg("shape"), py::arg("p")=0.5, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA, 
          "Generate binary random tensor");

    m.def("ones", &ones<float>, py::arg("shape"), 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("zeros", &zeros<float>, py::arg("shape"), 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("ones_like", &ones_like<float>, py::arg("input"));
    m.def("zeros_like", &zeros_like<float>, py::arg("input"));
    m.def("arange", &arange<float>, 
          py::arg("start"), 
          py::arg("end"), 
          py::arg("step")=1, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("constant", &constant<float>, py::arg("shape"), py::arg("val"),
                                        py::arg("dtype")=DataType::FLOAT,
                                        py::arg("device")=BackendType::CUDA);
    m.def("one_hot", &one_hot<float>, py::arg("size"), py::arg("idx"),
                                      py::arg("dtype")=DataType::FLOAT,
                                      py::arg("device")=BackendType::CUDA);
}

void bind_module_float(py::module &m) {
    py::module nn = m.def_submodule("nn", "Neural network operations");

    py::class_<Module<float>, std::shared_ptr<Module<float>>>(nn, "Module_float")
       .def(py::init<>())
       .def(py::init<std::vector<std::shared_ptr<Module<float>>> &>())
       .def("__call__", &Module<float>::operator())
       .def("train", &Module<float>::train)
       .def("eval", &Module<float>::eval)
       .def("parameters", &Module<float>::parameters)
       .def("forward", &Module<float>::forward);

    py::class_<Sequential<float>, Module<float>, 
                std::shared_ptr<Sequential<float>>>(nn, "Sequential_float")
        .def(py::init<std::vector<std::shared_ptr<Module<float>>> &>())
        .def("forward", &Sequential<float>::forward);

    py::class_<Linear<float>, Module<float>, 
                std::shared_ptr<Linear<float>>>(nn, "Linear_float")
        .def(py::init<int, int, bool, DataType, BackendType>(),
            py::arg("in_features"),
            py::arg("out_features"),
            py::arg("bias") = true,
            py::arg("dtype") = DataType::FLOAT,
            py::arg("device") = BackendType::CUDA)
        .def("forward", &Linear<float>::forward)
        .def("set_params", &Linear<float>::set_params,
            py::arg("params"),
            py::arg("dtype"),
            py::arg("device")=BackendType::CUDA)
        ;
}

