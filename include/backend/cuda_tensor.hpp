//#ifndef __CUDA_TENSOR_HPP__
//#define __CUDA_TENSOR_HPP__
#pragma once

#include "backend/base_tensor.hpp"
#include "backend/base_array.hpp"
#include "backend/cuda_backend.cuh"

namespace py = pybind11;

template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CudaTensor: public BaseTensor<Dtype> {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

    explicit CudaTensor(py::array_t<float>& np_array, DataType dtype=DataType::FLOAT);
    CudaTensor(const std::shared_ptr<GenericOp<Dtype>> op, 
               std::vector<cached_data_type> inputs): BaseTensor<Dtype>(op, inputs) {}
    explicit CudaTensor(const std::vector<int32_t>& shape, DataType dtype=DataType::FLOAT,
                        bool create_cache=true);
    ~CudaTensor() {}

    CudaTensor(const CudaTensor&)=delete;
    CudaTensor& operator=(const CudaTensor&)=delete;

    virtual void half(const float* data);
    virtual void to_float(float* data);
    virtual py::array_t<float> to_numpy() override;
    virtual void fill_val(Dtype val, DataType dtype) override;
    virtual void zeros() override;
    virtual void ones() override;
    virtual void arange(int32_t start, int32_t step, DataType dtype) override;
    virtual void from_buffer() override;
    inline virtual size_t size() override {
        return this->_prod(this->__shape);
    }
    virtual std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data() override;
    virtual inline BackendType device() override {return BackendType::CUDA;}

protected:
    virtual void _from_numpy(py::array_t<float> &a) override;
};

template<typename Dtype>
CudaTensor<Dtype>::CudaTensor(py::array_t<float>& np_array, DataType dtype): 
        BaseTensor<Dtype>(np_array, dtype) {

    size_t size = this->_prod(this->__shape);
    this->array.reset(new CudaArray<Dtype>(size));
    _from_numpy(np_array);
}

template<typename Dtype>
CudaTensor<Dtype>::CudaTensor(const std::vector<int32_t>& shape, 
                              DataType dtype, 
                              bool create_cache):
    BaseTensor<Dtype>(shape, dtype) {
    size_t size = this->_prod(this->__shape);
    this->array.reset(new CudaArray<Dtype>(size, create_cache));

    //std::cout << "selected cuda backend 2, " << create_cache << std::endl;
}

template<typename Dtype>
void CudaTensor<Dtype>::fill_val(Dtype val, DataType dtype) {
    this->array->fill_val(val);
    this->is_compact = true;
    this->cached = true;
    this->dtype = dtype;
}

template<typename Dtype>
void CudaTensor<Dtype>::zeros() {
    this->array->fill_val(static_cast<Dtype>(0));
    this->is_compact = true;
    this->cached = true;
}

template<typename Dtype>
void CudaTensor<Dtype>::arange(int32_t start, int32_t step, DataType dtype) {
    this->array->arange(static_cast<Dtype>(start), 
                        static_cast<Dtype>(step));

    this->is_compact = true;
    this->cached = true;
    this->dtype = dtype;
}

template<typename Dtype>
void CudaTensor<Dtype>::ones() {
    this->array->fill_val(static_cast<Dtype>(1));
    this->is_compact = true;
    this->cached = true;
}

template<typename Dtype>
void CudaTensor<Dtype>::from_buffer() {
    this->compact();
    // copy memory to host
    Dtype* host_ptr = (Dtype*)std::malloc(this->array->size() * sizeof(Dtype));
    if (host_ptr == 0) throw std::bad_alloc();

    this->array->mem_cpy(host_ptr, MemCpyType::Dev2Host);

    printf("[");
    for(size_t i=0; i<size(); ++i)
        printf("%f,", host_ptr[i]);

    printf("]\n");
}

template<typename Dtype>
void CudaTensor<Dtype>::half(const float* data) {
    this->array->half(data);
}

template<typename Dtype>
void CudaTensor<Dtype>::to_float(float* data) {
    this->array->to_float(data);
}

template<typename Dtype>
void CudaTensor<Dtype>::_from_numpy(py::array_t<float> &a) {
    const float* ptr = reinterpret_cast<const float*>(a.data());
    if constexpr (std::is_same_v<Dtype, float>) {
        this->array->mem_cpy(const_cast<float*>(ptr), MemCpyType::Host2Dev);
    } else if constexpr (std::is_same_v<Dtype, __half>) {
        auto host_fp32 = std::make_shared<CudaArray<float>>(this->array->size()*sizeof(float));
        host_fp32->mem_cpy(const_cast<float*>(ptr), MemCpyType::Host2Dev);
        half(host_fp32->get_ptr());
    }
}

template<typename Dtype>
py::array_t<float> CudaTensor<Dtype>::to_numpy() {

    std::vector<int32_t> numpy_strides = this->__strides;
    std::transform(numpy_strides.begin(), 
                   numpy_strides.end(), 
                   numpy_strides.begin(),
                   [](int32_t& c) { return c * sizeof(float); });

    // copy memory to host
    float* host_ptr = (float*)std::malloc(this->array->size() * sizeof(float));
    if (host_ptr == 0) throw std::bad_alloc();

    if constexpr (std::is_same_v<Dtype, float>) {
        this->array->mem_cpy(host_ptr, MemCpyType::Dev2Host);
    } else if constexpr (std::is_same_v<Dtype, __half>) {

        //std::shared_ptr<CudaArray<float>> d_fp32 = std::make_shared<
        auto d_fp32 = std::make_shared<CudaArray<float>>(this->array->size(), true); 
        to_float(d_fp32->get_ptr());
        d_fp32->mem_cpy(host_ptr, MemCpyType::Dev2Host);
    }

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<float>(this->__shape, numpy_strides, 
                              host_ptr + this->__offset, 
                              deallocate_buffer);
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> CudaTensor<Dtype>::deep_cpy_cached_data() {
    this->compact();
    std::shared_ptr<BaseTensor<Dtype>> cached_data = 
        std::make_shared<CudaTensor<Dtype>>(this->__shape);

    this->array->mem_cpy(cached_data->cached_ptr(), MemCpyType::Dev2Dev);

    return cached_data;
}

//#endif

