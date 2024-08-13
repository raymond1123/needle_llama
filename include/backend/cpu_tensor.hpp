//#ifndef __CPU_TENSOR_HPP__
//#define __CPU_TENSOR_HPP__
#pragma once

#include "backend/base_tensor.hpp"
#include "backend/base_array.hpp"
#include "backend/cpu_backend.hpp"

namespace py = pybind11;

//template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CpuTensor: public BaseTensor<Dtype> {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    explicit CpuTensor(const std::vector<int32_t>& shape, DataType dtype=DataType::FLOAT,
                       bool create_cache=true);
    CpuTensor(const std::shared_ptr<GenericOp<Dtype>> op, 
               std::vector<cached_data_type> inputs): BaseTensor<Dtype>(op, inputs) {}
    explicit CpuTensor(py::array_t<float>& np_array, DataType dtype=DataType::FLOAT);
    ~CpuTensor() {}

    CpuTensor(const CpuTensor&)=delete;
    CpuTensor& operator=(const CpuTensor&)=delete;

    virtual void half(const float* data) override;
    virtual void to_float(float* data);
    virtual py::array_t<float> to_numpy() override;
    virtual void fill_val(Dtype val, DataType dtype) override;
    virtual void zeros() override;
    virtual void arange(int32_t start, int32_t step, DataType dtype) override;
    virtual void ones() override;
    virtual void from_buffer() override;

    inline virtual size_t size() override {
        return this->_prod(this->__shape);
    }
    virtual std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data() override;

    virtual inline BackendType device() override {return BackendType::CPU;} 

protected:
    virtual void _from_numpy(py::array_t<float> &a) override;

/*
private:
    std::shared_ptr<CpuArray<Dtype>> array;
*/
};

template<typename Dtype>
CpuTensor<Dtype>::CpuTensor(py::array_t<float>& np_array, DataType dtype):
    BaseTensor<Dtype>(np_array, dtype) {
    size_t size = this->_prod(this->__shape);
    this->array.reset(new CpuArray<Dtype>(size));
    std::cout << "selected cpu backend" << std::endl;
    _from_numpy(np_array);
}

template<typename Dtype>
CpuTensor<Dtype>::CpuTensor(const std::vector<int32_t>& shape,
                            DataType dtype, 
                            bool create_cache):
    BaseTensor<Dtype>(shape, dtype) {
    size_t size = this->_prod(this->__shape);
    if(create_cache)
        this->array.reset(new CpuArray<Dtype>(size));

    std::cout << "selected cpu backend" << std::endl;
}

template<typename Dtype>
void CpuTensor<Dtype>::fill_val(Dtype val, DataType dtype) {
    this->array->fill_val(val);
    this->dtype = dtype;
}

template<typename Dtype>
void CpuTensor<Dtype>::zeros() {
    this->array->fill_val(static_cast<Dtype>(0));
}

template<typename Dtype>
void CpuTensor<Dtype>::arange(int32_t start, int32_t step, DataType dtype) {
    this->array->arange(static_cast<Dtype>(start), 
                        static_cast<Dtype>(step));
    this->dtype = dtype;
}

template<typename Dtype>
void CpuTensor<Dtype>::ones() {
    this->array->fill_val(static_cast<Dtype>(1));
}

template<typename Dtype>
void CpuTensor<Dtype>::from_buffer() {

    printf("[");
    for(size_t i=0; i<size(); ++i)
        printf("%f,", this->array->get_ptr()[i]);

    printf("]\n");

}

template<typename Dtype>
void CpuTensor<Dtype>::_from_numpy(py::array_t<float> &a) {
    const Dtype* ptr = reinterpret_cast<const Dtype*>(a.data());
    this->array->mem_cpy(const_cast<Dtype*>(ptr), 
                         MemCpyType::Host2Host);
}

template<typename Dtype>
py::array_t<float> CpuTensor<Dtype>::to_numpy() {
    if constexpr (std::is_same<Dtype, __half>::value) {
        throw std::runtime_error("to_numpy() is not supported for __half type.");
    } else {
        std::vector<int32_t> numpy_strides = this->__strides;
        std::transform(numpy_strides.begin(), 
                       numpy_strides.end(), 
                       numpy_strides.begin(),
                       [](int32_t& c) { return c * sizeof(Dtype); });

        return py::array_t<float>(this->__shape, numpy_strides, 
                                  this->cached_ptr() + this->__offset);
    }

    // return a empty numpy array
    return py::array_t<float>();
}

template<typename Dtype>
void CpuTensor<Dtype>::half(const float* data) {
    assert(true && "now on cpu, only backend on CUDA can convert fp32 to fp16");
}

template<typename Dtype>
void CpuTensor<Dtype>::to_float(float* data) {
    assert(true && "on cpu, data is already fp32");
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> CpuTensor<Dtype>::deep_cpy_cached_data() {
    this->compact();
    std::shared_ptr<BaseTensor<Dtype>> cached_data = 
        std::make_shared<CpuTensor<Dtype>>(this->__shape);

    this->array->mem_cpy(cached_data->cached_ptr(), MemCpyType::Hosta2Hostb);

    return cached_data;
}

//#endif

