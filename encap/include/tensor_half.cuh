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

void bind_operator_iplus_tensor(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__iadd__",
        [](Tensor<__half>& self, Tensor<__half>& other) {
            self += other;
            return self;
        });
}

void bind_operator_plus_tensor(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__add__",
        [](Tensor<__half>& self, Tensor<__half>& other) {
            return self + other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_plus_scalar(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__add__",
        [](Tensor<__half>& self, const __half scalar) {
            return self + scalar;
        });
}

void bind_operator_sub_tensor(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__sub__",
        [](Tensor<__half>& self, Tensor<__half>& other) {
            return self - other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_sub_scalar(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__sub__",
        [](Tensor<__half>& self, const __half scalar) {
            return self - scalar;
        });
}

void bind_operator_mul_tensor(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__mul__",
        [](Tensor<__half>& self, Tensor<__half>& other) {
            return self * other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_mul_scalar(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__mul__",
        [](Tensor<__half>& self, const __half scalar) {
            return self * scalar;
        });
}

void bind_operator_div_tensor(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__truediv__",
        [](Tensor<__half>& self, Tensor<__half>& other) {
            return self / other;
        });
}

// Binding for operator+ with a scalar
void bind_operator_div_scalar(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__truediv__",
        [](Tensor<__half>& self, const __half scalar) {
            return self / scalar;
        });
}

void bind_operator_pow_tensor(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__pow__",
        [](Tensor<__half>& self, Tensor<__half>& other) {
            return self.op_pow(other);
        });
}

void bind_operator_pow_scalar(py::class_<Tensor<__half>>& tensor_class) {
    tensor_class.def("__pow__",
        [](Tensor<__half>& self, const __half scalar) {
            return self.op_pow(scalar);
        });
}

void bind_tensor_half(py::module &m, const char *name) {

    py::class_<Tensor<__half>> tensor_class(m, name);
    tensor_class
        .def(py::init<py::array_t<float>&, DataType, BackendType>(),
            py::arg("np_array"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("backend")=BackendType::CUDA)

        .def(py::init<std::vector<int32_t>, DataType, BackendType>(),
            py::arg("shape"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("backend")=BackendType::CUDA)

        .def("reshape", &Tensor<__half>::reshape)
        .def("flip", &Tensor<__half>::flip)
        .def("__getitem__", &Tensor<__half>::slice)
        .def("__setitem__", &Tensor<__half>::setitem)
        .def("broadcast_to", &Tensor<__half>::broadcast_to)
        .def("permute", &Tensor<__half>::permute)
        .def("transpose", &Tensor<__half>::transpose)
        .def("sum", (Tensor<__half> (Tensor<__half>::*)(std::vector<int>)) &Tensor<__half>::summation, "Summation with specified axes")
        .def("sum", (Tensor<__half> (Tensor<__half>::*)()) &Tensor<__half>::summation, "Summation without specified axes")
        .def("max", &Tensor<__half>::max, py::arg("dim"), py::arg("keepdim")=false)
        .def("relu", &Tensor<__half>::relu)
        .def("tanh", &Tensor<__half>::tanh)
        .def("log", &Tensor<__half>::log)
        .def("exp", &Tensor<__half>::exp)
        .def("neg", &Tensor<__half>::neg)
        .def("__matmul__", &Tensor<__half>::matmul)
        .def("dilate", &Tensor<__half>::dilate, py::arg("dilation"), py::arg("axes"))
        .def("to_numpy", &Tensor<__half>::to_numpy)
        .def("device", &Tensor<__half>::device)
        .def("shape", &Tensor<__half>::shape)
        .def("size", &Tensor<__half>::size)
        .def("strides", &Tensor<__half>::strides)
        .def("offset", &Tensor<__half>::offset)
        .def("from_buffer", &Tensor<__half>::from_buffer)
        .def("contiguous", &Tensor<__half>::contiguous)
        .def("backward", &Tensor<__half>::backward)
        .def("grad", &Tensor<__half>::grad)
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

void bind_init_half(py::module &m) {
    m.def("rand", &rand<__half>, 
          py::arg("shape"), 
          py::arg("min")=0.0, 
          py::arg("max")=1.0, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA,
          "Generate uniformly distributed random tensor");

    m.def("randn", &randn<__half>, 
          py::arg("shape"), py::arg("mean")=0.0, 
          py::arg("std")=1.0, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA, 
          "Generate Gaussian distributed random tensor");

    m.def("randb", &randb<__half>, 
          py::arg("shape"), py::arg("p")=0.5, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA, 
          "Generate binary random tensor");

    m.def("ones", &ones<__half>, py::arg("shape"), 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("zeros", &zeros<__half>, py::arg("shape"), 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("ones_like", &ones_like<__half>, py::arg("input"));
    m.def("zeros_like", &zeros_like<__half>, py::arg("input"));
    m.def("arange", &arange<__half>, 
          py::arg("start"), 
          py::arg("end"), 
          py::arg("step")=1, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("constant", &constant<__half>, py::arg("shape"), py::arg("val"),
                                        py::arg("dtype")=DataType::FLOAT,
                                        py::arg("device")=BackendType::CUDA);
    m.def("one_hot", &one_hot<__half>, py::arg("size"), py::arg("idx"),
                                      py::arg("dtype")=DataType::FLOAT,
                                      py::arg("device")=BackendType::CUDA);
}


void bind_module_half(py::module &m) {
    py::module nn = m.def_submodule("nn", "Neural network operations");

    py::class_<Module<__half>, std::shared_ptr<Module<__half>>>(nn, "Module_half")
       .def(py::init<>())
       .def(py::init<std::vector<std::shared_ptr<Module<__half>>> &>())
       .def("__call__", &Module<__half>::operator())
       .def("train", &Module<__half>::train)
       .def("eval", &Module<__half>::eval)
       .def("parameters", &Module<__half>::parameters)
       .def("forward", &Module<__half>::forward);

    py::class_<Sequential<__half>, Module<__half>, 
                std::shared_ptr<Sequential<__half>>>(nn, "Sequential_half")
        .def(py::init<std::vector<std::shared_ptr<Module<__half>>> &>())
        .def("forward", &Sequential<__half>::forward);

    py::class_<Linear<__half>, Module<__half>, 
                std::shared_ptr<Linear<__half>>>(nn, "Linear_half")
        .def(py::init<int, int, bool, DataType, BackendType>(),
            py::arg("in_features"),
            py::arg("out_features"),
            py::arg("bias") = true,
            py::arg("dtype") = DataType::HALF,
            py::arg("device") = BackendType::CUDA)
        .def("forward", &Linear<__half>::forward)
        .def("set_params", &Linear<__half>::set_params,
            py::arg("params"),
            py::arg("dtype"),
            py::arg("device")=BackendType::CUDA)
        ;
}