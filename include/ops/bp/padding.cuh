#ifndef __PADDING_OP__
#define __PADDING_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/slice.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class PaddingOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    PaddingOp(OpType op_type, std::vector<int> axes, Dtype val): 
        GenericOp<Dtype>(op_type), _axes(axes) {}
        //_new_shape(std::vector<int32_t>(axes.size()/2)) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {
        assert(inputs.size() == 1 && "number of padding input must be 1");
        _org_shape = inputs[0]->shape();
        assert(_axes.size()%2==0 && "size of input axes must be even");
        assert(_axes.size() <= _org_shape.size()*2 && "input axes error");

        /* only accept positive axes */
        for(auto& a:_axes)
            assert(a >=  0 && "input axes error");

        calc_new_shape();
        cached_data_type cached_data = __create_cached_data(_new_shape,
                                                            inputs[0]->dtype,
                                                            inputs[0]->device());
        cached_data->zeros();

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _slices, 
                                             cached_data->shape(), 
                                             cached_data->strides(), 
                                             cached_data->offset());

        auto view = slice_op->compute({cached_data});

        _n = inputs[0]->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        ApplyEwSetitem<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  inputs[0]->cached_ptr(), 
                                                  view->cached_ptr(), // out
                                                  VecToCuda(view->shape()), 
                                                  VecToCuda(view->strides()), 
                                                  view->offset());

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEwSetitem failed");

        cached_data->cached = true;
        cached_data->is_compact = true;

        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {

        auto cached_out = __create_cached_data(_org_shape,
                                               out_grad->dtype,
                                               out_grad->device());

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _slices, 
                                             out_grad->shape(), 
                                             out_grad->strides(), 
                                             out_grad->offset());

        auto view = slice_op->compute({out_grad});

        _n = cached_out->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        ApplyEwSetitem<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  view->cached_ptr(), 
                                                  cached_out->cached_ptr(), // out
                                                  VecToCuda(cached_out->shape()), 
                                                  VecToCuda(cached_out->strides()), 
                                                  cached_out->offset());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEwSetitem failed");

        return {cached_out};
    }

private:
    void calc_new_shape() {
        int axes_length = _org_shape.size()*2;

        std::vector<int32_t> append_shape;
        _new_shape = _org_shape;
        std::vector<int32_t> tmp_axes;

        for (size_t i = _axes.size(); i > 0; i -= 2) {
            tmp_axes.push_back(_axes[i - 2]);
            tmp_axes.push_back(_axes[i - 1]);
        }
        _axes = tmp_axes; 
        _axes.insert(_axes.begin(), axes_length-_axes.size(), 0);

        for(int i=0; i<_axes.size(); i+=2)
            append_shape.push_back(_axes[i]+_axes[i+1]);

        for(int i=0; i<_org_shape.size(); ++i)
            _new_shape[i] = _org_shape[i]+append_shape[i];

        for(int i=0; i<_axes.size(); i+=2) {
            _slices.push_back(py::slice(_axes[i], _new_shape[i/2]-_axes[i+1], 1));
        }
    }

    cached_data_type __create_cached_data(const std::vector<int32_t>& shape, 
                                          DataType dtype,
                                          BackendType device,
                                          bool create_cache=true) {
        cached_data_type cached_data = nullptr;
        if (device == BackendType::CPU) {
            cached_data.reset(new CpuTensor<Dtype>(shape, dtype, create_cache));
        } else if (device == BackendType::CUDA) {
            cached_data.reset(new CudaTensor<Dtype>(shape, dtype, create_cache));
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }

        return cached_data;
    }

protected:
    virtual inline cudaError_t _get_num_blocks() override {
        _num_blocks = (_n + kBlockSize - 1) / kBlockSize;
        return cudaSuccess;
    }

private:
    std::vector<int32_t> _new_shape;
    std::vector<int32_t> _org_shape;
    std::vector<int32_t> _axes;
    size_t _n;
    int _num_blocks;
    std::vector<py::object> _slices;
};

#endif

