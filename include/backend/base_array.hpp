#ifndef __BASE_ARRAY_HPP__
#define __BASE_ARRAY_HPP__

#include "common.hpp"

template<typename Dtype>
class BaseArray {
public:
    using cached_array_type = std::shared_ptr<BaseArray<Dtype>>;

    BaseArray(const size_t size):__size(size), __ptr(nullptr) {};
    virtual ~BaseArray() {}

    BaseArray(const BaseArray&)=delete;
    BaseArray& operator=(const BaseArray&)=delete;

    inline size_t size() {return __size;}
    inline Dtype* get_ptr() {return __ptr;}
    inline void set_ptr(Dtype * ptr) {__ptr = ptr;}

    virtual void half(const float* data)=0;
    virtual void to_float(float* data)=0;
    virtual void arange(Dtype start, Dtype step)=0;
    virtual cached_array_type compact(size_t size, 
                                      std::vector<int32_t> shape,
                                      std::vector<int32_t> strides,
                                      size_t offset)=0;
    virtual void mem_cpy(Dtype* ptr, 
                         MemCpyType mem_cpy_type)=0;
    virtual void fill_val(Dtype val)=0;

protected:
    Dtype *__ptr;
    size_t __size;
};

#endif

