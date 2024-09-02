//#ifndef __TENSOR_HPP__
//#define __TENSOR_HPP__
#pragma once

#include <cuda_fp16.h>
#include "ops/ops_math.hpp"

namespace py = pybind11;

static int tensor_idx=0;

template<typename Dtype>
class Tensor {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

    /* constructor */
    Tensor() {}
    Tensor(py::array_t<float>& np_array, DataType dtype=DataType::FLOAT, 
           BackendType backend=BackendType::CUDA);
    Tensor(std::vector<int32_t> shape, DataType dtype=DataType::FLOAT, 
           BackendType backend=BackendType::CUDA);
    Tensor(DataType dtype,
           BackendType backend, 
           std::shared_ptr<GenericOp<Dtype>> op=nullptr,
           std::vector<cached_data_type> inputs={nullptr});

    static Tensor fill_val(std::vector<int32_t> shape, float val, 
                           DataType dtype=DataType::FLOAT,
                           BackendType backend=BackendType::CUDA);
    static Tensor ones(std::vector<int32_t> shape, 
                       DataType dtype=DataType::FLOAT,
                       BackendType backend=BackendType::CUDA);
    static Tensor zeros(std::vector<int32_t> shape, 
                        DataType dtype=DataType::FLOAT,
                        BackendType backend=BackendType::CUDA);
    static Tensor arange(int32_t start, int32_t end, int32_t step=1, 
                         DataType dtype=DataType::FLOAT,
                         BackendType backend=BackendType::CUDA);

    /* move/cpy constructor */
    Tensor(Tensor&& other) noexcept;
    Tensor(const Tensor& other);
    Tensor(Tensor* other);

    Tensor<__half> half();

    /* move/cpy operator= */
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data();

    inline cached_data_type cached_data() const { return __cached_data; }
    inline py::array_t<float> to_numpy() { return __cached_data->to_numpy(); }
    inline void from_buffer() { __cached_data->from_buffer(); }
    inline py::array_t<Dtype> grad() { return __cached_data->grad->to_numpy(); }
    inline std::vector<int32_t> shape() const { return __cached_data->shape(); }
    inline std::vector<int32_t> strides() { return __cached_data->strides(); }
    inline size_t offset() { return __cached_data->offset(); }
    //inline BackendType device() const {return __backend;}
    virtual inline size_t size() {return __cached_data->size();};

    inline Dtype* cached_ptr() {return __cached_data->cached_ptr();}

    inline void reset_cached_data(std::shared_ptr<BaseTensor<Dtype>> cached_data) {
        __cached_data = cached_data;
    }

    static Tensor make_from_op(const std::shared_ptr<GenericOp<Dtype>> op,
                               std::vector<cached_data_type>& inputs,
                               BackendType backend);

    static cached_data_type make_from_op_on_self(const std::shared_ptr<GenericOp<Dtype>> op,
                                          std::vector<cached_data_type>& inputs,
                                          BackendType backend,
                                          bool op_on_self);

    void realized_cached_data(const std::shared_ptr<GenericOp<Dtype>> op,
                              std::vector<cached_data_type>& inputs);

    /* operations */
    // element-wise addition
    Tensor operator+(Tensor& other);
    Tensor operator+(const Dtype scalar);
    Tensor operator-(Tensor& other);
    Tensor operator-(const Dtype scalar);
    Tensor operator*(Tensor& other);
    Tensor operator*(const Dtype scalar);
    Tensor operator/(Tensor& other);
    Tensor operator/(const Dtype scalar);
    Tensor op_pow(Tensor& other);
    Tensor op_pow(const Dtype scalar);

    Tensor matmul(const Tensor& other) const;
    Tensor embedding(const Tensor<float>& other);

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(Tensor&& other);

    Tensor reshape(std::vector<int32_t> new_shape);
    Tensor flip(std::vector<int> axes);
    Tensor broadcast_to(std::vector<int32_t> shape);
    Tensor slice(std::vector<py::object> indices); /* getitem */
    void setitem(std::vector<py::object> indices, Tensor& other);
    Tensor permute(std::vector<int> axes);
    Tensor transpose(std::vector<int> axes);
    Tensor summation(std::vector<int> axes);
    Tensor summation();
    std::vector<Tensor> max(int dim, bool keepdim);
    Tensor padding(std::vector<int> axes);
    Tensor dilate(uint32_t dilation, std::vector<int> axes);
    Tensor relu();
    Tensor silu();
    Tensor tanh();
    Tensor log();
    Tensor exp();
    Tensor neg();
    Tensor rms_norm();
    Tensor rotary_emb(int start_pos);
    Tensor softmax();

    /* backward */
    void backward();
    //void backward(std::shared_ptr<Tensor> out_grad);

    inline void contiguous() {__cached_data->compact();}

private:
    bool __check_type() {
        bool is_half=false;
        if constexpr (std::is_same_v<Dtype, __half>) is_half = true;
        if (is_half && device==BackendType::CPU) return false;
        return true;
    }

    void __print_tensor_info(std::string ctor_type);
    void __compute_gradient(cached_data_type out_tensor, 
                            cached_data_type out_grad);

    cached_data_type __sum_grad(std::vector<cached_data_type> input_grads);

    /* DFS for graph */
    void __find_topo_sort(std::vector<cached_data_type> tensors,
                          std::vector<cached_data_type>& reverse_topo_order);
    void __topo_sort_dfs(cached_data_type tensor_shptr, 
                std::unordered_map<cached_data_type, bool>& visited,
                std::vector<cached_data_type>& reverse_topo_order);

public:
    DataType dtype;
    BackendType device;

private:
    int __tensor_idx;

    std::shared_ptr<BaseTensor<Dtype>> __cached_data;
};

template<typename Dtype>
Tensor<Dtype>::Tensor(std::vector<int32_t> shape, DataType dtype, BackendType backend):
    dtype(dtype), device(backend) {

    if (device== BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            __cached_data = std::make_shared<CpuTensor<Dtype>>(shape, dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (device== BackendType::CUDA) {
        __cached_data = std::make_shared<CudaTensor<Dtype>>(shape, dtype);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __cached_data->tensor_idx = __tensor_idx;
    //__print_tensor_info("ctor");
    #endif
}

template<typename Dtype>
Tensor<Dtype>::Tensor(py::array_t<float>& np_array, 
                      DataType dtype,
                      BackendType backend): dtype(dtype), device(backend) {

    if (device== BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            __cached_data = std::make_shared<CpuTensor<Dtype>>(np_array, dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (device== BackendType::CUDA) {
        __cached_data = std::make_shared<CudaTensor<Dtype>>(np_array, dtype);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    __cached_data->cached = true;

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __cached_data->tensor_idx = __tensor_idx;
    //__print_tensor_info("plane ctor");
    #endif
}

template<typename Dtype>
Tensor<Dtype>::Tensor(DataType dtype, BackendType backend, 
           std::shared_ptr<GenericOp<Dtype>> op,
           std::vector<std::shared_ptr<BaseTensor<Dtype>>> inputs): 
           dtype(dtype), device(backend) {

    if (device== BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            __cached_data = std::make_shared<CpuTensor<Dtype>>(dtype, op, inputs);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (device== BackendType::CUDA) {
        __cached_data = std::make_shared<CudaTensor<Dtype>>(dtype, op, inputs);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    tensor_idx++;
    __tensor_idx = tensor_idx;
    __cached_data->tensor_idx = __tensor_idx;

    #ifdef DEBUG
    //__print_tensor_info("ctor");
    #endif
}

template<typename Dtype>
Tensor<__half> Tensor<Dtype>::half() {

    if constexpr (std::is_same_v<Dtype, float>) {
        Tensor<__half> half_tensor(__cached_data->shape(), DataType::HALF, BackendType::CUDA);
        half_tensor.cached_data()->half(this->__cached_data->cached_ptr(),
                                        this->__cached_data->cached);

        return half_tensor;
    } 

    throw std::runtime_error("cpu does not support half data type");
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::ones(std::vector<int32_t> shape,
                                  DataType dtype,
                                  BackendType backend) {

    Tensor<Dtype> tensor = Tensor<Dtype>(dtype, backend);

    if (backend == BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            tensor.__cached_data = std::make_shared<CpuTensor<Dtype>>(shape, dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (backend == BackendType::CUDA) {
        tensor.__cached_data = std::make_shared<CudaTensor<Dtype>>(shape, dtype);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    tensor.__cached_data->ones();
    tensor.__cached_data->cached = true;

    #ifdef DEBUG
    tensor_idx++;
    tensor.__tensor_idx = tensor_idx;
    tensor.__cached_data->tensor_idx = tensor.__tensor_idx;
    //tensor.__print_tensor_info("ones");
    #endif

    return tensor;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::arange(int32_t start, int32_t end, int32_t step, 
                                    DataType dtype,
                                    BackendType backend) {

    Tensor<Dtype> tensor = Tensor<Dtype>(dtype, backend);
    std::vector<int32_t> shape = {((end-start)+step-1)/step}; // ceil

    if (backend == BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            tensor.__cached_data = std::make_shared<CpuTensor<Dtype>>(shape, dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (backend == BackendType::CUDA) {
        tensor.__cached_data = std::make_shared<CudaTensor<Dtype>>(shape, dtype);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    tensor.__cached_data->arange(start, step, dtype);
    tensor.__cached_data->cached = true;

    #ifdef DEBUG
    tensor_idx++;
    tensor.__tensor_idx = tensor_idx;
    tensor.__cached_data->tensor_idx = tensor.__tensor_idx;
    #endif

    return tensor;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::zeros(std::vector<int32_t> shape,
                                   DataType dtype,
                                   BackendType backend) {
    Tensor<Dtype> tensor = Tensor<Dtype>(dtype, backend);

    if (backend == BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            tensor.__cached_data = std::make_shared<CpuTensor<Dtype>>(shape, dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (backend == BackendType::CUDA) {
        tensor.__cached_data = std::make_shared<CudaTensor<Dtype>>(shape, dtype);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    tensor.__cached_data->zeros();
    tensor.__cached_data->cached = true;

    #ifdef DEBUG
    tensor_idx++;
    tensor.__tensor_idx = tensor_idx;
    tensor.__cached_data->tensor_idx = tensor.__tensor_idx;
    //tensor.tensor_idx:__print_tensor_info("zeros");
    #endif

    return tensor;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::fill_val(std::vector<int32_t> shape, 
                                      float val, 
                                      DataType dtype,
                                      BackendType backend) {

    Tensor<Dtype> tensor = Tensor<Dtype>(dtype, backend);

    if (backend == BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            tensor.__cached_data = std::make_shared<CpuTensor<Dtype>>(shape, dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (backend == BackendType::CUDA) {
        tensor.__cached_data = std::make_shared<CudaTensor<Dtype>>(shape, dtype);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    tensor.__cached_data->fill_val(val, dtype);
    tensor.__cached_data->cached = true;

    #ifdef DEBUG
    tensor_idx++;
    tensor.__tensor_idx = tensor_idx;
    tensor.__cached_data->tensor_idx = tensor.__tensor_idx;
    #endif

    return tensor;
}

template<typename Dtype>
Tensor<Dtype>::Tensor(Tensor&& other) noexcept:
        __tensor_idx(other.__tensor_idx), dtype(other.dtype),
        device(other.device), 
        __cached_data(other.__cached_data) {

    __cached_data->tensor_idx = __tensor_idx;

    #ifdef DEBUG
    //printf("tensor_idx:%d, original=%p, new=%p, move constructor\n", 
    //       __tensor_idx, &other, this);
    //__print_tensor_info("move constructor");
    #endif
}

template<typename Dtype>
Tensor<Dtype>& Tensor<Dtype>::operator=(Tensor<Dtype>&& other) noexcept {

    if(this == &other) return *this;

    this->dtype = other.dtype;
    this->device = other.device;
    __tensor_idx = other.__tensor_idx;
    //device = other.device;
    __cached_data = other.__cached_data;
    __cached_data->tensor_idx = __tensor_idx;

    other.__cached_data = nullptr;

    #ifdef DEBUG
    //printf("tensor_idx:%d, move operator=\n", __tensor_idx);
    #endif

    return *this;
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Tensor& other): 
    __cached_data(other.__cached_data), device(other.device) {
    /*
    __cached_data = other.__cached_data->deep_cpy_cached_data();
    __cached_data->op = other.__cached_data->op;
    __cached_data->inputs = other.__cached_data->inputs;
    __cached_data->grad = other.__cached_data->grad;
    __cached_data->cached = other.__cached_data->cached;
    __cached_data->is_compact = other.__cached_data->is_compact;
    __cached_data->tensor_idx = other.__cached_data->tensor_idx;
    */

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __cached_data->tensor_idx = __tensor_idx;
    //__print_tensor_info("cpy ctor");
    //printf("tensor_idx:%d, cpy constructor\n", __tensor_idx);
    #endif
}

template<typename Dtype>
Tensor<Dtype>::Tensor(Tensor* other): 
    __cached_data(other->__cached_data), device(other->device) {
    /*
    __cached_data = other.__cached_data->deep_cpy_cached_data();
    __cached_data->op = other.__cached_data->op;
    __cached_data->inputs = other.__cached_data->inputs;
    __cached_data->grad = other.__cached_data->grad;
    __cached_data->cached = other.__cached_data->cached;
    __cached_data->is_compact = other.__cached_data->is_compact;
    __cached_data->tensor_idx = other.__cached_data->tensor_idx;
    */

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __cached_data->tensor_idx = __tensor_idx;
    //__print_tensor_info("cpy ctor");
    //printf("tensor_idx:%d, cpy constructor\n", __tensor_idx);
    #endif
}

template<typename Dtype>
Tensor<Dtype>& Tensor<Dtype>::operator=(const Tensor<Dtype>& other) {
    if(this==&other) return *this;

    this->dtype = other.dtype;
    device = other.device;

    //__cached_data = other.__cached_data->deep_cpy_cached_data();
    //__cached_data->op = other.__cached_data->op;
    //__cached_data->inputs = other.__cached_data->inputs;
    __cached_data = other.__cached_data;

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __cached_data->tensor_idx = __tensor_idx;
    //__print_tensor_info("cpy operator=");
    //printf("tensor_idx:%d, cpy operator=\n", __tensor_idx);
    #endif

    return *this;
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> Tensor<Dtype>::deep_cpy_cached_data() {
    return __cached_data->deep_cpy_cached_data();
}

// this = this + other
template<typename Dtype>
Tensor<Dtype>& Tensor<Dtype>::operator+=(const Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.device == device&& 
           "backend of operators must be the same");

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWAddTensor);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);
    //printf("===============+\n");

    __cached_data = (*op)(op, inputs, device, true);

    return *this;
}

template<typename Dtype>
Tensor<Dtype>& Tensor<Dtype>::operator-=(Tensor<Dtype>&& other) {
    //check same beakend 
    assert(other.device == device && 
           "backend of operators must be the same");

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMinusTensor);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);
    //printf("===============+\n");

    __cached_data = (*op)(op, inputs, device, true);

    return *this;
}

// return = this + other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator+(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.device == device && 
           "backend of operators must be the same");

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWAddTensor);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator+(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWAddScalar,
                                      scalar);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

// return = this - other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator-(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.device==device && 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMinusTensor);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator-(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMinusScalar,
                                      scalar);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

// return = this * other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator*(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.device==device && 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMulTensor);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator*(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMulScalar,
                                      scalar);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

// return = this * other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator/(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.device==device&& 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWDivTensor);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator/(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWDivScalar,
                                      scalar);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

// return = this * other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::op_pow(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.device==device&& 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWPowTensor);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::op_pow(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWPowScalar,
                                      scalar);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::matmul(const Tensor<Dtype>& other) const {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<MatMulOp<Dtype>>(OpType::MatMul);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    inputs.push_back(other.__cached_data);

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::embedding(const Tensor<float>& index) {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EmbeddingOp<Dtype>>(index.cached_data(), 
                                             OpType::Embedding);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //inputs.push_back(other.__cached_data);

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::reshape(std::vector<int32_t> new_shape) {
    assert(__check_type() && "cpu does not support half data type");
    int neg_axis = -1;
    size_t input_size = 1;

    for(int i=0; i<new_shape.size(); ++i) {
        if(new_shape[i]==-1) neg_axis = i; 
        else input_size *= new_shape[i];
    }

    if(neg_axis != -1)
        new_shape[neg_axis] = size()/input_size;

    //assert(new_size==__cached_data->size() && "reshape input new_shape is not legal");

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<ReshapeOp<Dtype>>(new_shape, OpType::Reshape);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::flip(std::vector<int> axes) {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<FlipOp<Dtype>>(OpType::Flip, axes);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::broadcast_to(std::vector<int32_t> shape) {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<BroadcastOp<Dtype>>(shape, OpType::BroadcastTo);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::permute(std::vector<int> axes) {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<PermuteOp<Dtype>>(axes, OpType::Permute);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::transpose(std::vector<int> axes) {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<TransposeOp<Dtype>>(axes, OpType::Transpose);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::summation(std::vector<int> axes) {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<SummationOp<Dtype>>(axes, OpType::Summation);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::summation() {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<SummationOp<Dtype>>(OpType::Summation);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::rms_norm() {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<RMSNormOp<Dtype>>(OpType::RMSNorm);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::rotary_emb(int start_pos) {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<RotaryEmbOp<Dtype>>(start_pos, OpType::RotaryEmb);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::softmax() {

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<SoftmaxOp<Dtype>>(OpType::Softmax);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
std::vector<Tensor<Dtype>> Tensor<Dtype>::max(int dim, bool keepdim) {
    /* calc output shape */
    std::vector<int32_t> shape = __cached_data->shape(); 
    std::vector<int32_t> out_shape; 

    dim = dim<0?shape.size()+dim:dim;
    for(int i=0; i<__cached_data->shape().size(); ++i) {
        if(keepdim) {
            if(i==dim) out_shape.push_back(1);
            else out_shape.push_back(shape[i]);
        } else {
            if(i==dim) continue;
            else out_shape.push_back(shape[i]);
        }
    }

    auto idx = Tensor<Dtype>::zeros(out_shape, __cached_data->dtype, device);

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<MaxOp<Dtype>>(OpType::Max, dim, idx.__cached_data, keepdim);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    Tensor<Dtype> out = (*op)(op, inputs, device);

    return {out, idx};
}

/*
   The padding axes by which to pad some dimensions of input are described starting from the last dimension and moving forward (the same as pytorch)
 */
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::padding(std::vector<int> axes) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<PaddingOp<Dtype>>(axes, OpType::Padding);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::dilate(uint32_t dilation, std::vector<int> axes) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<DilateOp<Dtype>>(OpType::Dilate, axes, dilation);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::relu() {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<ReluOp<Dtype>>(OpType::Relu);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::silu() {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<SiluOp<Dtype>>(OpType::Silu);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::tanh() {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<TanhOp<Dtype>>(OpType::Tanh);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::log() {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<LogOp<Dtype>>(OpType::Log);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::exp() {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<ExpOp<Dtype>>(OpType::Exp);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::neg() {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<NegOp<Dtype>>(OpType::Neg);

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::slice(std::vector<py::object> indices) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<SliceOp<Dtype>>(OpType::Slice, indices, 
                                         shape(), strides(), offset());

    std::vector<cached_data_type> inputs;
    inputs.push_back(__cached_data);
    //printf("===============+\n");

    return (*op)(op, inputs, device);
}

template<typename Dtype>
void Tensor<Dtype>::setitem(std::vector<py::object> indices, Tensor& other) {

    // move *this to tmp
    cached_data_type tmp_cached_data = __cached_data;

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<SetitemOp<Dtype>>(OpType::Setitem, indices);

    std::vector<cached_data_type> inputs;
    inputs.push_back(tmp_cached_data);
    inputs.push_back(other.__cached_data);

    // assign __cached_dat a new pointer
    if (device== BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            __cached_data.reset(new CpuTensor<Dtype>(DataType::FLOAT, op, inputs));
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (device== BackendType::CUDA) {
        __cached_data.reset(new CudaTensor<Dtype>(DataType::FLOAT, op, inputs));
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    __cached_data = __cached_data->realized_cached_data();
    __cached_data->op = op;
    __cached_data->inputs = inputs;

    #ifdef DEBUG
    tensor_idx++;
    __cached_data->tensor_idx = __tensor_idx;
    #endif
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::make_from_op(const std::shared_ptr<GenericOp<Dtype>> op,
                                          std::vector<cached_data_type>& inputs,
                                          BackendType backend) {

    assert(inputs.size() > 0 && "number of inputs is zero");

    Tensor<Dtype> new_t = Tensor<Dtype>(inputs[0]->dtype, backend, op, inputs);

    new_t.__cached_data = new_t.__cached_data->realized_cached_data();

    //new_t.dtype = inputs[0]->dtype;
    //new_t.device = inputs[0]->device;

    new_t.__cached_data->op = op;
    new_t.__cached_data->inputs = inputs;

    return new_t;
}

template<typename Dtype>
void Tensor<Dtype>::realized_cached_data(const std::shared_ptr<GenericOp<Dtype>> op,
                               std::vector<std::shared_ptr<BaseTensor<Dtype>>>& inputs) {

    __cached_data = __cached_data->realized_cached_data();
    __cached_data->op = op;
    __cached_data->inputs = inputs;
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> Tensor<Dtype>::make_from_op_on_self(
                            const std::shared_ptr<GenericOp<Dtype>> op,
                            std::vector<cached_data_type>& inputs,
                            BackendType backend,
                            bool op_on_self) {

    assert(inputs.size() > 0 && "number of inputs is zero");
    cached_data_type cached_data = nullptr;

    if (backend == BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            cached_data = std::make_shared<CpuTensor<Dtype>>(inputs[0]->dtype, 
                                                             op, inputs);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (backend == BackendType::CUDA) {
        cached_data = std::make_shared<CudaTensor<Dtype>>(inputs[0]->dtype,
                                                          op, inputs);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    cached_data = cached_data->realized_cached_data();
    cached_data->op = op;
    cached_data->inputs = inputs;

    return cached_data;
}

template<typename Dtype>
void Tensor<Dtype>::backward() {
    __print_tensor_info("backward");

    std::shared_ptr<Tensor<Dtype>> out_grad = 
        std::make_shared<Tensor<Dtype>>(device);

    if (device== BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            out_grad->__cached_data = 
                    std::make_shared<CpuTensor<Dtype>>(__cached_data->shape(), 
                                                       __cached_data->dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (device== BackendType::CUDA) {
        out_grad->__cached_data = 
                    std::make_shared<CudaTensor<Dtype>>(__cached_data->shape(),
                                                        __cached_data->dtype);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    out_grad->__cached_data->ones();

    __compute_gradient(__cached_data, out_grad->__cached_data);
}

template<typename Dtype>
void Tensor<Dtype>::__compute_gradient(cached_data_type out_tensor, 
                                       cached_data_type out_grad) {

    using grad_type = std::shared_ptr<Tensor<Dtype>>;
    std::unordered_map<cached_data_type, std::vector<cached_data_type>> grad_map;

    grad_map[out_tensor] = {out_grad};
    std::vector<cached_data_type> reverse_topo_order;

    __find_topo_sort({out_tensor}, reverse_topo_order);
    std::reverse(reverse_topo_order.begin(), reverse_topo_order.end());

    #ifdef DEBUG
    for(auto& tensor: reverse_topo_order) {
        int op_type = -1;
        if (tensor->op != nullptr) {
            op_type = tensor->op->op_type();
        } 

        printf("tensor_idx=%d, op=%d, addr=%p-->\n", 
               tensor->tensor_idx, op_type, &tensor);
    }
    printf("\n");
    #endif

    for(auto& tensor: reverse_topo_order) {

        tensor->grad = __sum_grad(grad_map[tensor]);
        if(tensor->op!=nullptr) {
            std::vector<cached_data_type> grads;

            grads = tensor->op->gradient(tensor->grad, tensor);

            for(int i=0; i<tensor->inputs.size(); ++i) {
                grad_map[tensor->inputs[i]].push_back(grads[i]);
            }
        }
    }
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> Tensor<Dtype>::__sum_grad(
            std::vector<std::shared_ptr<BaseTensor<Dtype>>> input_grads) {
    assert(input_grads.size() > 0 && "at least one input gradients");

    // create grad Tensor
    cached_data_type grad = nullptr;
    if (device== BackendType::CPU) {
        if constexpr (std::is_same_v<Dtype, float>) {
            grad = std::make_shared<CpuTensor<Dtype>>(input_grads[0]->shape(), 
                                                      input_grads[0]->dtype);
        } else {
            throw std::runtime_error("cpu does not support half data type");
        }
    } else if (device== BackendType::CUDA) {
        grad = std::make_shared<CudaTensor<Dtype>>(input_grads[0]->shape(),
                                                   input_grads[0]->dtype);
    }

    grad->zeros();

    for(auto &in_grad: input_grads) {
        std::shared_ptr<EWOp<Dtype>> op =
                 std::make_shared<EWOp<Dtype>>(grad->size(), 
                                               OpType::EWAddTensor, 0, true);
        std::vector<cached_data_type> cin = {grad, in_grad};
        op->compute(cin);
    }

    return grad;
}

template<typename Dtype>
void Tensor<Dtype>::__find_topo_sort(
                std::vector<std::shared_ptr<BaseTensor<Dtype>>> tensors,
                std::vector<std::shared_ptr<BaseTensor<Dtype>>>& reverse_topo_order) {
    std::unordered_map<cached_data_type, bool> visited;

    for (auto& tensor_shptr: tensors) {
        if(!visited[tensor_shptr])
            __topo_sort_dfs(tensor_shptr, visited, reverse_topo_order);
    }
}

template<typename Dtype>
void Tensor<Dtype>::__topo_sort_dfs(
                std::shared_ptr<BaseTensor<Dtype>> tensor_shptr, 
                std::unordered_map<std::shared_ptr<BaseTensor<Dtype>>, bool>& visited,
                std::vector<std::shared_ptr<BaseTensor<Dtype>>>& reverse_topo_order) {

    visited[tensor_shptr] = true;
    for(auto& input: tensor_shptr->inputs) {
        if(!visited[input]) {
            visited[input] = true;
            __topo_sort_dfs(input, visited, reverse_topo_order);
        }
    }

    reverse_topo_order.push_back(tensor_shptr);
}

template<typename Dtype>
void Tensor<Dtype>::__print_tensor_info(std::string ctor_type) {
    printf("------tensor info---------\n");
    printf("%s:", ctor_type.c_str());
    std::vector<int> inputs_idx;
    int num_inputs = __cached_data->inputs.size();
    inputs_idx.reserve(num_inputs);

    printf("tensor_idx=%d; ", __tensor_idx);
    printf("tensor_addr=%p; ", this);

    if(num_inputs > 0) {
        printf("; inputs_addr= ");
        for(auto &e: __cached_data->inputs) {
            printf("%p, ", &e);
        }

    } else {
        printf("inputs_idx=None; ");
    }

    if(__cached_data->op!=nullptr)
        printf("op: %d\n", static_cast<int>(__cached_data->op->_op_type));
    else
        printf("op: nullptr\n");
    printf("----tensor info done------\n\n");
}

//#endif

