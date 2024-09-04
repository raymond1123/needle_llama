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
            dtype(other.dtype), device(other.device), __tensor(other.__tensor) {}

    // move ctor
    NdlTensor(const NdlTensor&& other) noexcept:
            dtype(other.dtype), device(other.device), 
            __tensor(std::move(other.__tensor)) {}

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

        this->dtype = other.dtype;
        this->device = other.device;
        this->__tensor = std::move(other.__tensor);

        return *this;
    }


    inline py::array_t<float> to_numpy() {
        return std::visit([](auto& tensor) -> py::array_t<float> {
            return tensor.to_numpy();
        }, __tensor);
    }

    inline std::vector<int32_t> shape() const { 
        return std::visit([](auto& tensor) -> std::vector<int32_t> {
            return tensor.shape();
        }, __tensor);
    }

    inline std::vector<int32_t> strides() { 
        return std::visit([](auto& tensor) -> std::vector<int32_t> {
            return tensor.strides();
        }, __tensor);
    }

    // operator+ for adding another NdlTensor
    NdlTensor operator+(NdlTensor& other) {
        return std::visit([&](auto& lhs, auto& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                //auto result = lhs + rhs;
                auto result = NdlTensor(lhs+rhs);
                result.dtype = lhs.dtype;
                result.device = lhs.device;
                return result;
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
            //auto result = tensor + scalar;
            auto result = NdlTensor(tensor+scalar);
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;

        }, this->__tensor);
    }

    // operator- for adding another NdlTensor
    NdlTensor operator-(NdlTensor& other) {
        return std::visit([&](auto&& lhs, auto&& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                //auto result = lhs - rhs;
                auto result = NdlTensor(lhs-rhs);
                result.dtype = lhs.dtype;
                result.device = lhs.device;

                return result;
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
            //auto result = tensor - scalar;
            auto result = NdlTensor(tensor - scalar);
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    // operator* for adding another NdlTensor
    NdlTensor operator*(NdlTensor& other) {
        return std::visit([&](auto& lhs, auto& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                //auto result = lhs * rhs;
                auto result = NdlTensor(lhs*rhs);
                result.dtype = lhs.dtype;
                result.device = lhs.device;
                return result;
            } else {
                throw std::invalid_argument("Tensor types must match");
            }
            return *this;
        }, this->__tensor, other.__tensor);
    }

    // operator* for adding a scalar
    template<typename ScalarType>
    NdlTensor operator*(ScalarType scalar) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            //auto result = tensor * scalar;
            auto result = NdlTensor(tensor*scalar);
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    // operator/ for adding another NdlTensor
    NdlTensor operator/(NdlTensor& other) {
        return std::visit([&](auto&& lhs, auto&& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                //auto result = lhs / rhs;
                auto result = NdlTensor(lhs/rhs);
                result.dtype = lhs.dtype;
                result.device = lhs.device;
                return result;
            } else {
                throw std::invalid_argument("Tensor types must match");
            }
            return *this;
        }, this->__tensor, other.__tensor);
    }

    // operator/ for adding a scalar
    template<typename ScalarType>
    NdlTensor operator/(ScalarType scalar) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            //auto result = tensor / scalar;
            auto result = NdlTensor(tensor/scalar);
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
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

    static NdlTensor where(const NdlTensor& condition,
                           const NdlTensor& x, const NdlTensor& y) {
        return std::visit([&](const auto& condition, 
                              const auto& x, const auto& y) -> NdlTensor {

            using CondType = std::decay_t<decltype(condition)>;
            using xType = std::decay_t<decltype(x)>;
            using yType = std::decay_t<decltype(y)>;

            if constexpr (std::is_same_v<CondType, Tensor<float>> &&
                          std::is_same_v<xType, Tensor<float>> &&
                          std::is_same_v<yType, Tensor<float>>) {
                auto result = NdlTensor(f_where(condition, x, y));
                result.dtype = y.dtype;
                result.device = y.device;
                return result;
            } else if constexpr (std::is_same_v<CondType, Tensor<float>> &&
                                 std::is_same_v<xType, Tensor<__half>> &&
                                 std::is_same_v<yType, Tensor<__half>>) {
                auto result = NdlTensor(f_where(condition, x, y));
                result.dtype = y.dtype;
                result.device = y.device;
                return result;
            }

            throw std::invalid_argument("Unsupported DataType");
        }, condition.__tensor, x.__tensor, y.__tensor);

        return NdlTensor(y.dtype, y.device);
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
    NdlTensor matmul(const NdlTensor& other) const {
        return std::visit([&](const auto& lhs, const auto& rhs) -> NdlTensor {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;

            if constexpr (std::is_same_v<LhsType, RhsType>) {
                auto result = NdlTensor(lhs.matmul(rhs));
                result.dtype = rhs.dtype;
                result.device = rhs.device;
                return result;
            } else {
                throw std::invalid_argument("Tensor types must match");
            }
            return *this;
        }, this->__tensor, other.__tensor);
    }

    NdlTensor summation(const std::vector<int>& axes) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            //return tensor.summation(axes);
            auto result = NdlTensor(tensor.summation(axes));
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    NdlTensor rms_norm(const NdlTensor& weight) {
        return std::visit([&](auto& tensor, const auto& weight) -> NdlTensor {
            using LhsType = std::decay_t<decltype(tensor)>;
            using RhsType = std::decay_t<decltype(weight)>;

            // 确保只有相同类型的 Tensor 能相加
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                auto result = NdlTensor(tensor.rms_norm(weight));
                result.dtype = tensor.dtype;
                result.device = tensor.device;
                return result;
            } else {
                throw std::invalid_argument("Tensor types must match");
            }
            return *this;
        }, this->__tensor, weight.__tensor);
    }

    NdlTensor rotary_emb(int start_pos) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = NdlTensor(tensor.rotary_emb(start_pos));
            result.dtype = tensor.dtype;
            result.device = tensor.device;

            return result;
        }, this->__tensor);
    }

    NdlTensor slice(std::vector<py::object> indices) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = NdlTensor(tensor.slice(indices));
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    void setitem(std::vector<py::object> indices, NdlTensor& other) {
        std::visit([&](auto& self, auto& tensor) {
            if constexpr (std::is_same_v<std::decay_t<decltype(self)>, 
                          std::decay_t<decltype(tensor)>>) {
                self.setitem(indices, tensor);
            } else {
                throw std::runtime_error("Mismatched tensor types in setitem.");
            }
        }, this->__tensor, other.__tensor);
    }

    NdlTensor embedding(const NdlTensor& index) {
        return std::visit([&](auto& tensor, const auto& index) -> NdlTensor {
            using RhsType = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<RhsType, Tensor<float>>) {
                auto result = NdlTensor(tensor.embedding(index));
                result.dtype = tensor.dtype;
                result.device = tensor.device;
                return result;
            } else {
                throw std::invalid_argument("currently, embedding only support float index");
            }

            return *this;
        }, this->__tensor, index.__tensor);
    }

    NdlTensor softmax() {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = NdlTensor(tensor.softmax());
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    std::pair<NdlTensor, NdlTensor> argmax(int dim, bool keepdim) {

        if (std::holds_alternative<Tensor<float>>(__tensor)) {
            auto tensor = std::get<Tensor<float>>(__tensor);
            auto [idx, out] = tensor.argmax(dim, keepdim);
            return { NdlTensor(idx), NdlTensor(out) };
        } else if (std::holds_alternative<Tensor<__half>>(__tensor)) {
            auto tensor = std::get<Tensor<__half>>(__tensor);
            auto [idx, out] = tensor.argmax(dim, keepdim);
            return { NdlTensor(idx), NdlTensor(out) };
        } else {
            throw std::runtime_error("Unsupported tensor type");
        }
    }

    NdlTensor summation() {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = NdlTensor(tensor.summation());
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }
    
    NdlTensor transpose(std::vector<int> axes) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = NdlTensor(tensor.transpose(axes));
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    NdlTensor reshape(std::vector<int32_t> new_shape) {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = NdlTensor(tensor.reshape(new_shape));
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    void contiguous() {
        std::visit([&](auto& tensor) {
            tensor.contiguous();
        }, this->__tensor);
    }

    NdlTensor silu() {
        return std::visit([&](auto& tensor) -> NdlTensor {
            auto result = NdlTensor(tensor.silu());
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;

        }, this->__tensor);
    }

    NdlTensor broadcast_to(std::vector<int32_t> shape) {

        return std::visit([&](auto& tensor) {
            using T = std::decay_t<decltype(tensor)>;
            // Broadcast this_tensor to the specified shape and assign it to result.__tensor
            auto result = NdlTensor(tensor.broadcast_to(shape));
            result.dtype = tensor.dtype;
            result.device = tensor.device;
            return result;
        }, this->__tensor);
    }

    NdlTensor half() {
        NdlTensor result;

        std::visit([&](auto& tensor) {
            using T = std::decay_t<decltype(tensor)>;
            if constexpr (std::is_same_v<T, Tensor<float>>) {
                result.__tensor = tensor.half();
                result.dtype = DataType::HALF;
                result.device = tensor.device;
            }
        }, this->__tensor);

        return result;
    }

    NdlTensor to_float() {
        NdlTensor result;

        std::visit([&](auto& tensor) {
            using T = std::decay_t<decltype(tensor)>;
            if constexpr (std::is_same_v<T, Tensor<__half>>) {
                result.__tensor = tensor.to_float();
                result.dtype = DataType::FLOAT;
                result.device = tensor.device;
            }
        }, this->__tensor);

        return result;
    }

    void from_buffer() {
        std::visit([&](auto& tensor) {
            tensor.from_buffer();
        }, this->__tensor);
    }

public:
    DataType dtype;
    BackendType device;

    NdlTensor(const Tensor<float>& tensor) : __tensor(tensor) {}
    NdlTensor(const Tensor<__half>& tensor) : __tensor(tensor) {}

private:
    std::variant<Tensor<float>, Tensor<__half>> __tensor;

    template <typename Dtype>
    NdlTensor(const Tensor<Dtype>& tensor) {
        __tensor.emplace<Tensor<Dtype>>(tensor);
    }
};

