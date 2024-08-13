#ifndef __SPLIT_OP__
#define __SPLIT_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/setitem.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class SplitOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    SplitOp(int dim, int idx, OpType op_type):
        GenericOp<Dtype>(op_type), _dim(dim), _idx(idx), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of SplitOp must be 1");

        //inputs[0]->compact();
        _org_shape = inputs[0]->shape();
        int split_size = _org_shape[_dim];
        __calc_new_shapes(split_size);

        cached_data_type cached_out = __create_cached_data(_split_shape,
                                                           inputs[0]->dtype,
                                                           inputs[0]->device());

        _n = cached_out->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        __split_tensor(cached_out, inputs[0]);

        cached_out->cached = true;
        cached_out->is_compact= true;

        return cached_out;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {

        cached_data_type cached_out = __create_cached_data(_org_shape,
                                                           out_grad->dtype,
                                                           out_grad->device());
        cached_out->zeros();
        std::vector<py::object> indices;

        int split_idx=0;
        for(int j=0; j<_org_shape.size(); ++j) {
            if(j==_dim) indices.push_back(py::slice(_idx, _idx+1,1));
            else {
                indices.push_back(py::slice(0, _split_shape[split_idx], 1));
                split_idx++;
            }
        }

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, indices, 
                                             cached_out->shape(), 
                                             cached_out->strides(), 
                                             cached_out->offset());

        auto view = slice_op->compute({cached_out});
        //view->compact();

        _n = view->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        ApplyEwSetitem<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  out_grad->cached_ptr(), 
                                                  view->cached_ptr(), // out
                                                  VecToCuda(view->shape()), 
                                                  VecToCuda(view->strides()), 
                                                  view->offset());

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEwSetitem failed");

        cached_out->compact(); // compact here before reduced add

        return {cached_out};
    }

protected:

    virtual inline cudaError_t _get_num_blocks() override {
        int dev, sm_count, tpm;
        cudaError err = __get_gpu_info(&dev, &sm_count, &tpm);
        _num_blocks = std::max<int>(1, std::min<int64_t>((_n + kBlockSize - 1) / kBlockSize,
                                               sm_count * tpm / kBlockSize * NUMWAVES));
        return cudaSuccess;
    }

private:
    void __split_tensor(cached_data_type cached_out, cached_data_type input) {

        std::vector<py::object> indices;

        int split_idx=0;
        for(int j=0; j<_org_shape.size(); ++j) {
            if(j==_dim) indices.push_back(py::slice(_idx, _idx+1,1));
            else {
                indices.push_back(py::slice(0, _split_shape[split_idx], 1));
                split_idx++;
            }
        }

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, indices, 
                                             input->shape(), 
                                             input->strides(), 
                                             input->offset());

        auto view = slice_op->compute({input});
        view->compact();

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
    }

    void __calc_new_shapes(int split_size) {

        int shape_length = _org_shape.size();

        if(_dim<0) _dim += shape_length;
        assert((0<=_dim && _dim<shape_length) && "dim not in the right scope");

        for(int i=0; i<shape_length; ++i)
            if(i!=_dim) _split_shape.push_back(_org_shape[i]);
    }

    inline cudaError_t __get_gpu_info(int* dev, int* sm_count, int* tpm) {
        cudaError_t err = cudaGetDevice(dev);
        if (err != cudaSuccess) { return err; }
        err = cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, *dev);
        if (err != cudaSuccess) { return err; }
        err = cudaDeviceGetAttribute(tpm, cudaDevAttrMaxThreadsPerMultiProcessor, *dev);
        if (err != cudaSuccess) { return err; }
        return cudaSuccess;
    }

    inline cached_data_type __create_cached_data(const std::vector<int32_t>& shape, 
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

private:
    size_t _n;
    int _num_blocks;

    int _dim;
    int _idx;
    std::vector<int32_t> _org_shape;
    std::vector<int32_t> _split_shape;
};

#endif

