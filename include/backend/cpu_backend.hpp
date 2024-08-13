//#ifndef __CPU_BACKEND_HPP__
//#define __CPU_BACKEND_HPP__

#pragma once

#include "common.hpp"

#define ALIGNMENT 256
namespace py = pybind11;

enum compact_mode {COMPACT, EWITEM, SITEM};

template<typename Dtype>
class CpuArray: public BaseArray<Dtype> {
public:
    using cached_array_type = std::shared_ptr<BaseArray<Dtype>>;

    CpuArray(const size_t size, bool cached_data=true);
    ~CpuArray() {free(this->__ptr);}

    CpuArray(const CpuArray&)=delete;
    CpuArray& operator=(const CpuArray&)=delete;

    void mem_cpy(Dtype* ptr,
                 MemCpyType mem_cpy_type) override;

    virtual void half(const float* data) override;
    virtual void to_float(float* data) override;
    virtual void fill_val(Dtype val) override;
    virtual void arange(Dtype start, Dtype step) override;
    virtual cached_array_type compact(size_t size, 
                              std::vector<int32_t> shape,
                              std::vector<int32_t> strides,
                              size_t offset) override;

private:
    void __deep_cpy(Dtype* other_ptr);
    void __assign_value(compact_mode mode,
                        const Dtype* a, 
                        Dtype* out, 
                        std::vector<int32_t> &shape,
                        std::vector<int32_t> &strides, 
                        size_t offset,
                        std::vector<int> &indices,
                        int depth,
                        size_t &idx,
                        Dtype val);
};

template<typename Dtype>
CpuArray<Dtype>::CpuArray(const size_t size, bool create_cache): 
    BaseArray<Dtype>(size) {

    if(create_cache) {
        int ret = posix_memalign(reinterpret_cast<void**>(&(this->__ptr)),
                             ALIGNMENT, this->__size*sizeof(Dtype));

        if (ret != 0) throw std::bad_alloc();
    }
}

template<typename Dtype>
void CpuArray<Dtype>::mem_cpy(Dtype* ptr, MemCpyType mem_cpy_type) {
    if(mem_cpy_type==MemCpyType::Host2Host)
        std::memcpy(this->__ptr, ptr, this->__size*sizeof(Dtype));
    else if(mem_cpy_type==MemCpyType::Hosta2Hostb)
        __deep_cpy(ptr);
}

template<typename Dtype>
void CpuArray<Dtype>::__deep_cpy(Dtype* other_ptr) {
    for(size_t i=0; i<this->__size; ++i)
        other_ptr[i] = this->__ptr[i];
}

template<typename Dtype>
void CpuArray<Dtype>::arange(Dtype start, Dtype step) {
    this->__ptr[0] = start;
    for(size_t i=1; i<this->__size; ++i)
        this->__ptr[i] += step;
}

template<typename Dtype>
void CpuArray<Dtype>::fill_val(Dtype val) {
    if(val==static_cast<Dtype>(0)) {
        memset(this->__ptr, val, this->__size*sizeof(Dtype));
    } else {
        for(size_t i=0; i<this->__size; ++i)
            this->__ptr[i] = val;
    }
}

template<typename Dtype>
void CpuArray<Dtype>::half(const float* data) {
    assert(true && "now is on cpu, only on CUDA can convert fp32 to fp16");
}

template<typename Dtype>
void CpuArray<Dtype>::to_float(float* data) {
    assert(true && "on cpu, data is already fp32");
}

/* this is actually a DFS recursive */
template<typename Dtype>
void CpuArray<Dtype>::__assign_value(compact_mode mode,
                                    const Dtype* a, 
                                    Dtype* out, 
                                    std::vector<int32_t> &shape,
                                    std::vector<int32_t> &strides, 
                                    size_t offset,
                                    std::vector<int> &indices,
                                    int depth,
                                    size_t &idx,
                                    Dtype val) {

    // Base case: reached the innermost loop 
    // leaf node in recursive tree
    if (depth == shape.size()) {
        size_t in_idx = 0;
        for(int i=0; i<shape.size(); ++i) {
            in_idx += strides[i]*indices[i];
        }

        if(mode == COMPACT) 
            out[idx++] = a[in_idx+offset];
        if(mode == EWITEM) 
            out[in_idx+offset] = a[idx++];
        if(mode == SITEM) 
            out[in_idx+offset] = val;

        return;

    } else { // Recursive case: iterate over the current dimension
        for(int i=0; i<shape[depth]; ++i) {
            indices[depth] = i;
            __assign_value(mode, a, out, shape, strides, offset, 
                      indices, depth+1, idx, val);
        }
    }
}

template<typename Dtype>
std::shared_ptr<BaseArray<Dtype>> CpuArray<Dtype>::compact(size_t size, 
                                           std::vector<int32_t> shape,
                                           std::vector<int32_t> strides,
                                           size_t offset) {
    cached_array_type array = std::make_shared<CpuArray<Dtype>>(size);

    std::vector<int> indices(shape.size(), 0);  
    size_t in_idx = 0;
    __assign_value(COMPACT, 
                   this->__ptr, 
                   array->get_ptr(), 
                   shape, strides, offset,
                   indices, 0, in_idx, -1);

    return array;
}

//#endif

