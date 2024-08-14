#pragma once
#include "tensor.cuh"

namespace py = pybind11;

class NdlTensor {
public:
    /* constructor */
    NdlTensor() {}
    NdlTensor(py::array_t<float>& np_array, 
              DataType dtype=DataType::FLOAT, 
              BackendType backend=BackendType::CUDA) {

        if (dtype == DataType::FLOAT) {
            __tensor.emplace<Tensor<float>>(np_array, dtype, backend);
        } else if (dtype == DataType::HALF) {
            __tensor.emplace<Tensor<__half>>(np_array, dtype, backend);
        } else {
            throw std::invalid_argument("Unsupported DataType");
        }
    }

    // to_numpy 方法
    inline py::array_t<float> to_numpy() {
        return std::visit([](auto& tensor) -> py::array_t<float> {
            return tensor.to_numpy();
        }, __tensor);
    }

    inline std::vector<int32_t> shape() { 
        return std::visit([](auto& tensor) -> std::vector<int32_t> {
            return tensor.shape();
        }, __tensor);
    }

    inline std::vector<int32_t> strides() { 
        return std::visit([](auto& tensor) -> std::vector<int32_t> {
            return tensor.strides();
        }, __tensor);
    }

    inline BackendType device() const {
        return std::visit([](auto& tensor) -> BackendType {
            return tensor.device();
        }, __tensor);
    }

    // operator+ for adding another NdlTensor
    NdlTensor operator+(NdlTensor& other) {
        return std::visit([&](auto& lhs, auto& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                auto result = lhs + rhs;
                return NdlTensor(result);
            } else {
                throw std::invalid_argument("Tensor types must match");
            }
            return *this;
        }, this->__tensor, other.__tensor);
    }

    // operator+ for adding a scalar
    template<typename ScalarType>
    NdlTensor operator+(ScalarType scalar) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = tensor + scalar;
            return NdlTensor(result);
        }, this->__tensor);
    }

public:
    DataType dtype;

private:
    std::variant<Tensor<float>, Tensor<__half>> __tensor;

    template <typename Dtype>
    NdlTensor(const Tensor<Dtype>& tensor) {
        __tensor.emplace<Tensor<Dtype>>(tensor);
    }
};

