#ifndef __DILATE_OP__
#define __DILATE_OP__

#include "ops/generic_op.cuh"
#include "backend/cuda_util.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class DilateOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    DilateOp(OpType op_type, std::vector<int> axes, uint32_t dilation):
        GenericOp<Dtype>(op_type), _axes(axes), _dilation(dilation) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size() == 1 && "number of reshape input must be 1");
        inputs[0]->compact();

        __calc_new_shape(inputs[0]->shape());

        cached_data_type cached_out = __create_cached_data(_new_shape,
                                                           inputs[0]->dtype,
                                                           inputs[0]->device());
        cached_out->zeros();

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _slices, 
                                             cached_out->shape(), 
                                             cached_out->strides(), 
                                             cached_out->offset());

        auto view = slice_op->compute({cached_out});

        assert(view->shape().size()==inputs[0]->shape().size() && "slice in dilate failed");
        for(int i=0; i<view->shape().size(); ++i) 
            assert(view->shape()[i]==inputs[0]->shape()[i] && "slice in dilate failed");

        _n = view->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        ApplyEwSetitem<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  inputs[0]->cached_ptr(), 
                                                  view->cached_ptr(), // out
                                                  VecToCuda(view->shape()), 
                                                  VecToCuda(view->strides()), 
                                                  view->offset());
        cached_out->cached = true;
        cached_out->is_compact = true;

        return cached_out;
    }

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor) override {

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _slices, 
                                             out_grad->shape(), 
                                             out_grad->strides(), 
                                             out_grad->offset());
        /*
        for(auto& l: _slices)
            printf("llllllll: %d,%d,%d\n", int(py::int_(l.attr("start"))),
                                           int(py::int_(l.attr("stop"))),
                                           int(py::int_(l.attr("step"))));
        */

        auto cached_out = slice_op->compute({out_grad});
        cached_out->compact();

        return {cached_out};
    }

private:
    void __calc_new_shape(std::vector<int32_t> org_shape) {
        _new_shape = org_shape;

        for(auto& idx: _axes)
            _new_shape[idx] *= (_dilation+1);

        for(auto& s: _new_shape)
            _slices.push_back(py::slice(0,s,1));

        for(auto& idx: _axes)
            _slices[idx] = py::slice(0, _new_shape[idx], _dilation+1);

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

protected:
    inline cudaError_t __get_gpu_info(int* dev, int* sm_count, int* tpm) {
        cudaError_t err = cudaGetDevice(dev);
        if (err != cudaSuccess) { return err; }
        err = cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, *dev);
        if (err != cudaSuccess) { return err; }
        err = cudaDeviceGetAttribute(tpm, cudaDevAttrMaxThreadsPerMultiProcessor, *dev);
        if (err != cudaSuccess) { return err; }
        return cudaSuccess;
    }

    virtual inline cudaError_t _get_num_blocks() override {
        //int dev, sm_count, tpm;
        //cudaError err = __get_gpu_info(&dev, &sm_count, &tpm);
        //_num_blocks = std::max<int>(1, std::min<int64_t>((_n + kBlockSize - 1) / kBlockSize,
        //                                       sm_count * tpm / kBlockSize * NUMWAVES));
        _num_blocks = (_n + kBlockSize - 1) / kBlockSize;
        return cudaSuccess;
    }

private:
    size_t _n;
    int _num_blocks;

    std::vector<int> _axes;
    std::vector<int32_t> _new_shape;
    std::vector<py::object> _slices;
    uint32_t _dilation;

};

#endif

