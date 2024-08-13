#ifndef __SETITEM_OP__
#define __SETITEM_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/slice.cuh"
#include "backend/cuda_util.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class SetitemOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    SetitemOp(OpType op_type, std::vector<py::object>& indices):
        GenericOp<Dtype>(op_type), _indices(indices) {}

    // inputs[0] ==> self
    // inputs[1] ==> other
    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==2 && "input number of SetitemOp must be 1");
        // TODO inputs[1] should not be compact
        inputs[1]->compact();

        auto out_cached = inputs[0]->deep_cpy_cached_data();
        //_n = out_cached->size();

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _indices, 
                                             out_cached->shape(), 
                                             out_cached->strides(), 
                                             out_cached->offset());

        auto view = slice_op->compute({out_cached});

        _n = view->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        // TODO inputs[1] should not be compact
        ApplyEwSetitem<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  inputs[1]->cached_ptr(), 
                                                  view->cached_ptr(), // out
                                                  VecToCuda(view->shape()), 
                                                  VecToCuda(view->strides()), 
                                                  view->offset());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEwSetitem failed");

        out_cached->cached = true;
        out_cached->is_compact= true;

        return out_cached;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {

        auto out_cached1 = out_grad->deep_cpy_cached_data();

        _n = out_cached1->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _indices, 
                                             out_cached1->shape(), 
                                             out_cached1->strides(), 
                                             out_cached1->offset());

        auto view = slice_op->compute({out_cached1});
        err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        ApplyEwSetscalar<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n, static_cast<Dtype>(0),
                                                  view->cached_ptr(), // out
                                                  VecToCuda(view->shape()), 
                                                  VecToCuda(view->strides()), 
                                                  view->offset());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEwSetscalar failed");

        std::shared_ptr<GenericOp<Dtype>> slice_op2 = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _indices, 
                                            out_grad->shape(), 
                                            out_grad->strides(), 
                                            out_grad->offset());

        auto out_cached2 = slice_op2->compute({out_grad});
        out_cached2->compact();

        return {out_cached1, out_cached2};
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
    std::vector<Py_ssize_t> __process_slice(py::slice slice, size_t length) {
        Py_ssize_t start, stop, step, slicelength;
        if (!slice.compute(length, &start, &stop, &step, &slicelength)) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to extract slice indices");
            throw std::runtime_error("Failed to extract slice indices");
            return {};
        }

        return {start, stop, step, slicelength};
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
                                                 BackendType device,
                                                 bool create_cache=true) {
        cached_data_type cached_data = nullptr;
        if (device == BackendType::CPU) {
            cached_data.reset(new CpuTensor<Dtype>(shape, create_cache));
        } else if (device == BackendType::CUDA) {
            cached_data.reset(new CudaTensor<Dtype>(shape, create_cache));
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }

        return cached_data;
    }

private:
    size_t _n;
    int _num_blocks;
    std::vector<py::object> _indices;
};

#endif

