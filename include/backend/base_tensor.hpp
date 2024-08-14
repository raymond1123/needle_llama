#ifndef __BASE_TENSOR_HPP__
#define __BASE_TENSOR_HPP__

#include "common.hpp"
#include "backend/base_array.hpp"
#include "ops/generic_op.cuh"

namespace py = pybind11;

template<typename Dtype>
class BaseTensor {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    using cached_array_type = std::shared_ptr<BaseArray<Dtype>>;

    BaseTensor(const std::vector<int32_t> &shape, DataType dtype=DataType::FLOAT);
    BaseTensor(py::array_t<float> &np_array, DataType dtype=DataType::FLOAT);

    BaseTensor(const std::shared_ptr<GenericOp<Dtype>> op, 
               std::vector<cached_data_type> inputs): 
        op(op), inputs(inputs), cached(false), is_compact(true) {}

    virtual ~BaseTensor()=default;

    virtual py::array_t<float> to_numpy()=0;
    virtual inline size_t size()=0;
    virtual void fill_val(Dtype val, DataType dtype)=0;
    virtual void half(const float* data)=0;
    virtual void to_float(float* data)=0;
    virtual void zeros()=0;
    virtual void ones()=0;
    virtual void arange(int32_t start, int32_t step, DataType dtype)=0;
    virtual void from_buffer()=0;
    virtual BackendType device()=0;
    virtual std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data()=0;

    inline std::vector<int32_t> shape() {return __shape;}
    inline std::vector<int32_t> strides() {return __strides;}
    inline size_t offset() {return __offset;}

    void set_shape(std::vector<int32_t> new_shape) {
        __shape = new_shape;
        compact_strides();
    }

    inline void set_strides(std::vector<int32_t> new_strides) {__strides = new_strides;}
    inline void set_offset(size_t offset) {__offset = offset;}

    inline Dtype* cached_ptr() { return this->array->get_ptr();}
    inline void set_cached_ptr(Dtype* ptr) { array->set_ptr(ptr); }

    /* input cdata is shared_ptr of itself*/
    cached_data_type realized_cached_data(cached_data_type cdata=nullptr);

    void compact_strides();
    void compact();

protected:
    virtual void _from_numpy(py::array_t<float> &np_array)=0;

    size_t _prod(const std::vector<int32_t> &shape);
    void _get_info_from_numpy(py::array_t<float> &np_array);

public:
    DataType dtype;
    cached_array_type array;
    std::shared_ptr<GenericOp<Dtype>> op;
    std::vector<cached_data_type> inputs;
    cached_data_type grad;
    bool cached;
    bool is_compact;
    int tensor_idx;

protected:
    py::dtype __dtype;
    std::vector<int32_t> __shape;
    std::vector<int32_t> __strides;
    size_t __offset;
    bool __cached;
};

template<typename Dtype>
BaseTensor<Dtype>::BaseTensor(const std::vector<int32_t>& shape, DataType dtype):
    __shape(shape), __offset(0), dtype(dtype),
    __strides(std::vector<int32_t>(shape.size())), 
    cached(false), is_compact(true) {
    compact_strides();
}

template<typename Dtype>
BaseTensor<Dtype>::BaseTensor(py::array_t<float>& np_array, DataType dtype): 
    __offset(0), cached(false), is_compact(true), dtype(dtype) {
    _get_info_from_numpy(np_array);
}

template<typename Dtype>
size_t BaseTensor<Dtype>::_prod(const std::vector<int32_t>& shape) {
    size_t size = 1;
    for (auto &s: __shape)
        size *= s;
    return size;
}

template<typename Dtype>
void BaseTensor<Dtype>::compact_strides() {
    __strides[__strides.size()-1] = 1;

    /* 
     * do not need to check whether strides.size()==1
     * the for loop does the job
     */
    for(int i=__strides.size()-2; i>=0; --i) {
        __strides[i] = __strides[i+1]*__shape[i+1];
    }
}

template<typename Dtype>
void BaseTensor<Dtype>::_get_info_from_numpy(py::array_t<float>& np_array) {
    assert(np_array.dtype().is(py::dtype::of<float>()) && "numpy array must be float");

    py::buffer_info buf_info = np_array.request();

    for (auto dim : buf_info.shape)
        __shape.push_back(dim);

    for (auto stride : buf_info.strides)
        __strides.push_back(stride / sizeof(float));
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> BaseTensor<Dtype>::realized_cached_data(
                                      std::shared_ptr<BaseTensor<Dtype>> cdata) {
    if(cached) return cdata;

    std::vector<cached_data_type> cin;
    for (int i=0; i<inputs.size(); ++i)
        cin.emplace_back(inputs[i]->realized_cached_data(inputs[i]));

    return op->compute(cin);
}

template<typename Dtype>
void BaseTensor<Dtype>::compact() {
    if(is_compact) return;

    size_t out_size = _prod(__shape);
    auto new_array = array->compact(out_size, __shape, __strides, __offset);
    array = new_array;

    compact_strides();
    __offset=0;
    is_compact = true;
}

#endif

