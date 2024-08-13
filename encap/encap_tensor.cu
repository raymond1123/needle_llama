#include "encap_tensor.cuh"

PYBIND11_MODULE(needle, m) {
    py::enum_<DataType>(m, "DataType")
        .value("FLOAT", DataType::FLOAT)
        .value("HALF", DataType::HALF)
        .value("INT8", DataType::INT8)
        .value("INT4", DataType::INT4)
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

    /* tensor */
    bind_tensor_float(m, "TensorFloat");
    bind_tensor_half(m, "TensorHalf");

    m.def("stack_float", &stack<float>, py::arg("inputs"), py::arg("dim")=0);
    m.def("split_float", &split<float>, py::arg("input"), py::arg("dim")=0);

    m.def("stack_half", &stack<__half>, py::arg("inputs"), py::arg("dim")=0);
    m.def("split_half", &split<__half>, py::arg("input"), py::arg("dim")=0);

    /* nn.functional */
    bind_functional(m);

    /* nn.init */
    bind_init_float(m);
    bind_init_half(m);

    /* nn.module */
    bind_module_float(m);
    bind_module_half(m);
}
