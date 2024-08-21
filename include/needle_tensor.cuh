#pragma once

#include "tensor.cuh"
#include "nn/function.cuh"

namespace py = pybind11;

class NdlTensor {
public:
    /* constructor */
    NdlTensor() {}
    NdlTensor(py::array_t<float>& np_array, 
              DataType dtype=DataType::FLOAT, 
              BackendType backend=BackendType::CUDA): dtype(dtype), device(backend) {

        if (dtype == DataType::FLOAT) {
            __tensor.emplace<Tensor<float>>(np_array, dtype, backend);
        } else if (dtype == DataType::HALF) {
            __tensor.emplace<Tensor<__half>>(np_array, dtype, backend);
        } else {
            throw std::invalid_argument("Unsupported DataType");
        }
    }

    NdlTensor(DataType dtype, BackendType backend) {
        if (dtype == DataType::FLOAT) {
            __tensor.emplace<Tensor<float>>(dtype, backend);
        } else if (dtype == DataType::HALF) {
            __tensor.emplace<Tensor<__half>>(dtype, backend);
        } else {
            throw std::invalid_argument("Unsupported DataType");
        }
    }

    // cpy ctor
    NdlTensor(const NdlTensor& other): 
            dtype(other.dtype), device(other.device), __tensor(other.__tensor) {

        std::visit([&](auto& tensor, const auto& other) {
            tensor.dtype = other.dtype; 
        }, this->__tensor, other.__tensor);

        printf("aaaaaa\n");
    }

    // move ctor
    NdlTensor(const NdlTensor&& other) noexcept:
            dtype(other.dtype), device(other.device), __tensor(other.__tensor) {

        std::visit([&](auto& tensor) {
            tensor.dtype = other.dtype; 
        }, __tensor);

        printf("bbbbbbb\n");
    }

    // cpy op= 
    NdlTensor& operator=(const NdlTensor& other) {
        if (this == &other) return *this;

        this->dtype = other.dtype;
        this->device = other.device;
        this->__tensor = other.__tensor;

        return *this;
    }

    // move op=
    NdlTensor& operator=(NdlTensor&& other) noexcept {
        if (this == &other) return *this;

        this->dtype = std::move(other.dtype);
        this->device = std::move(other.device);
        this->__tensor = std::move(other.__tensor);

        return *this;
    }


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

    //inline BackendType device() const {
    //    return std::visit([](auto& tensor) -> BackendType {
    //        return tensor.device();
    //    }, __tensor);
    //}

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

    // operator+= for adding another NdlTensor
    NdlTensor& operator+=(const NdlTensor& other) {
        std::visit([&](auto& lhs, auto& rhs) {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                lhs += rhs;
            } else {
                throw std::invalid_argument("Tensor types must match");
            }

        }, this->__tensor, other.__tensor);

        return *this;
    }

    // operator+ for adding a scalar
    template<typename ScalarType>
    NdlTensor operator+(ScalarType scalar) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = tensor + scalar;
            return NdlTensor(result);
        }, this->__tensor);
    }

    // operator- for adding another NdlTensor
    NdlTensor operator-(NdlTensor& other) {
        return std::visit([&](auto&& lhs, auto&& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                auto result = lhs - rhs;
                return NdlTensor(result);
            } else {
                throw std::invalid_argument("Tensor types must match");
            }
            return *this;
        }, this->__tensor, other.__tensor);
    }

    // operator+ for adding a scalar
    template<typename ScalarType>
    NdlTensor operator-(ScalarType scalar) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = tensor - scalar;
            return NdlTensor(result);
        }, this->__tensor);
    }

    static NdlTensor arange(int32_t start, int32_t end, int32_t step=1,
                     DataType dtype=DataType::FLOAT,
                     BackendType backend=BackendType::CUDA) {

        NdlTensor tensor  = NdlTensor(dtype, backend);
        
        if (dtype == DataType::FLOAT) {
            tensor.__tensor.emplace<Tensor<float>>(
                Tensor<float>::arange(start, end, step, dtype, backend));
        } else if (dtype == DataType::HALF) {
            tensor.__tensor.emplace<Tensor<__half>>(
                Tensor<__half>::arange(start, end, step, dtype, backend));
        } else {
            throw std::invalid_argument("Unsupported DataType");
        }

        return std::move(tensor);
    }

    static NdlTensor ones(std::vector<int32_t> shape,
                          DataType dtype,
                          BackendType backend) {
        NdlTensor tensor  = NdlTensor(dtype, backend);

        if (dtype == DataType::FLOAT) {
            tensor.__tensor.emplace<Tensor<float>>(
                Tensor<float>::ones(shape, dtype, backend));
        } else if (dtype == DataType::HALF) {
            tensor.__tensor.emplace<Tensor<__half>>(
                Tensor<__half>::ones(shape, dtype, backend));
        } else {
            throw std::invalid_argument("Unsupported DataType");
        }

        return tensor;
    }

    static NdlTensor zeros(std::vector<int32_t> shape, 
                    DataType dtype, BackendType backend) {
        NdlTensor tensor  = NdlTensor(dtype, backend);

        if (dtype == DataType::FLOAT) {
            tensor.__tensor.emplace<Tensor<float>>(
                Tensor<float>::zeros(shape, dtype, backend));
        } else if (dtype == DataType::HALF) {
            tensor.__tensor.emplace<Tensor<__half>>(
                Tensor<__half>::zeros(shape, dtype, backend));
        } else {
            throw std::invalid_argument("Unsupported DataType");
        }

        return tensor;
    }

    static NdlTensor fill_val(std::vector<int32_t> shape, float val, 
                       DataType dtype, BackendType backend) {

        NdlTensor tensor  = NdlTensor(dtype, backend);

        if (dtype == DataType::FLOAT) {
            tensor.__tensor.emplace<Tensor<float>>(
                Tensor<float>::fill_val(shape, val, dtype, backend));
        } else if (dtype == DataType::HALF) {
            tensor.__tensor.emplace<Tensor<__half>>(
                Tensor<__half>::fill_val(shape, val, dtype, backend));
        } else {
            throw std::invalid_argument("Unsupported DataType");
        }

        return tensor;
    }

    // matmul another NdlTensor
    NdlTensor matmul(const NdlTensor& other) {
        return std::visit([&](auto& lhs, auto& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            if constexpr (std::is_same_v<LhsType, RhsType>) {
                auto result = lhs.matmul(rhs);
                return NdlTensor(result);
            } else {
                throw std::invalid_argument("Tensor types must match");
            }
            return *this;
        }, this->__tensor, other.__tensor);
    }

    NdlTensor summation(const std::vector<int>& axes) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            return tensor.summation(axes);
        }, this->__tensor);
    }

    NdlTensor rms_norm() {
        return std::visit([&](auto& tensor) -> NdlTensor {
            return tensor.rms_norm();
        }, this->__tensor);
    }

    NdlTensor summation() {
        return std::visit([&](auto& tensor) -> NdlTensor {
            return tensor.summation();
        }, this->__tensor);
    }
    
    NdlTensor transpose(std::vector<int> axes) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            return tensor.transpose(axes);
        }, this->__tensor);
    }

    NdlTensor broadcast_to(std::vector<int32_t> shape) {
        NdlTensor result;

        std::visit([&](auto& tensor) {
            using T = std::decay_t<decltype(tensor)>;
            // Broadcast this_tensor to the specified shape and assign it to result.__tensor
            result.__tensor = tensor.broadcast_to(shape);
        }, this->__tensor);

        return result;
    }

    NdlTensor half() {
        NdlTensor result;

        std::visit([&](auto& tensor) {
            using T = std::decay_t<decltype(tensor)>;
            if constexpr (std::is_same_v<T, Tensor<float>>) {
                result.__tensor = tensor.half();
                result.dtype = DataType::HALF;
                result.device = tensor.device();
            }
        }, this->__tensor);

        return result;
    }

public:
    DataType dtype;
    BackendType device;

private:
    std::variant<Tensor<float>, Tensor<__half>> __tensor;

    template <typename Dtype>
    NdlTensor(const Tensor<Dtype>& tensor) {
        __tensor.emplace<Tensor<Dtype>>(tensor);
    }
};

