#include "tensor_float.cuh"
#include "tensor_half.cuh"

namespace py = pybind11;

void bind_functional(py::module &m) {
    py::module nn = m.def_submodule("nn", "Neural network operations");
    py::module functional = nn.def_submodule("functional", "Functions used in neural networks");
}

template<typename Dtype>
void bind_init(py::module &m) {
    m.def("rand", &rand<Dtype>, 
          py::arg("shape"), py::arg("min")=0.0, 
          py::arg("max")=1.0, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA,
          "Generate uniformly distributed random tensor");

    m.def("randn", &randn<Dtype>, 
          py::arg("shape"), py::arg("mean")=0.0, 
          py::arg("std")=1.0, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA, 
          "Generate Gaussian distributed random tensor");

    m.def("randb", &randb<Dtype>, 
          py::arg("shape"), py::arg("p")=0.5, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA, 
          "Generate binary random tensor");

    m.def("ones", &ones<Dtype>, py::arg("shape"), 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("zeros", &zeros<Dtype>, py::arg("shape"), 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("ones_like", &ones_like<Dtype>, py::arg("input"));
    m.def("zeros_like", &zeros_like<Dtype>, py::arg("input"));
    m.def("arange", &arange<Dtype>, 
          py::arg("start"), 
          py::arg("end"), 
          py::arg("step")=1, 
          py::arg("dtype")=DataType::FLOAT,
          py::arg("device")=BackendType::CUDA);

    m.def("constant", &constant<Dtype>, py::arg("shape"), py::arg("val"),
                                        py::arg("dtype")=DataType::FLOAT,
                                        py::arg("device")=BackendType::CUDA);
    m.def("one_hot", &one_hot<Dtype>, py::arg("size"), py::arg("idx"),
                                      py::arg("dtype")=DataType::FLOAT,
                                      py::arg("device")=BackendType::CUDA);
}

template<typename Dtype>
void bind_module(py::module &m) {
    py::module nn = m.def_submodule("nn", "Neural network operations");

    py::class_<Module<Dtype>, std::shared_ptr<Module<Dtype>>>(nn, "Module")
       .def(py::init<>())
       .def(py::init<std::vector<std::shared_ptr<Module<Dtype>>> &>())
       .def("__call__", &Module<Dtype>::operator())
       .def("train", &Module<Dtype>::train)
       .def("eval", &Module<Dtype>::eval)
       .def("parameters", &Module<Dtype>::parameters)
       .def("forward", &Module<Dtype>::forward);

    py::class_<Sequential<Dtype>, Module<Dtype>, 
                std::shared_ptr<Sequential<Dtype>>>(nn, "Sequential")
        .def(py::init<std::vector<std::shared_ptr<Module<Dtype>>> &>())
        .def("forward", &Sequential<Dtype>::forward);

    py::class_<Linear<Dtype>, Module<Dtype>, 
                std::shared_ptr<Linear<Dtype>>>(nn, "Linear")
        .def(py::init<int, int, bool, BackendType>(),
            py::arg("in_features"),
            py::arg("out_features"),
            py::arg("bias") = true,
            py::arg("device") = BackendType::CUDA)
        .def("forward", &Linear<Dtype>::forward)
        .def("set_params", &Linear<Dtype>::set_params,
            py::arg("params"),
            py::arg("dtype"),
            py::arg("device")=BackendType::CUDA)
        ;
}
