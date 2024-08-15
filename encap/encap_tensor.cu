#include <cuda_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "needle_tensor.cuh"
#include "needle_util.cuh"
#include "nn/function.cuh"
#include "nn/nn_module.cuh"
#include "nn/linear.cuh"
#include "init/init_basic.cuh"
#include "init/initial.hpp"

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
}

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

    py::class_<NdlTensor> tensor_class(m, "Tensor");
    tensor_class
        .def(py::init<py::array_t<float>&, DataType, BackendType>(),
            py::arg("np_array"),
            py::arg("dtype")=DataType::FLOAT,
            py::arg("backend")=BackendType::CUDA)

        .def("to_numpy", &NdlTensor::to_numpy)
        .def("device", &NdlTensor::device)
        .def("shape", &NdlTensor::shape)
        .def("strides", &NdlTensor::strides)
        .def("matmul", &NdlTensor::matmul)


        .def("__add__", [](NdlTensor& a, NdlTensor& b) {
            return a + b;
        }, py::is_operator())

        .def("__add__", [](NdlTensor& a, float scalar) {
            return a + scalar;
        }, py::is_operator())

        // 反向的 operator+，处理 float + NdlTensor
        .def("__radd__", [](NdlTensor& a, float scalar) {
            return a + scalar;
        }, py::is_operator())

        // 使用运算符重载 @ 运算符
        .def("__matmul__", &NdlTensor::matmul)

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

    bind_function(m);

    m.attr("fp32") = DataType::FLOAT;
    m.attr("fp16") = DataType::HALF;
    m.attr("cuda") = BackendType::CUDA;
    m.attr("cpu") = BackendType::CPU;
}


