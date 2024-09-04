#ifndef __WHERE_OP__
#define __WHERE_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class WhereKernel {
public:
    __device__ void operator()(size_t n,
                               const float* condition, 
                               const Dtype* x, 
                               const Dtype* y, 
                               Dtype* out) {

        size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
        if(tx>=n) return;

        if(condition[tx]==1.0) out[tx] = x[tx];
        else out[tx] = y[tx];
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyWhere(size_t n, const float* condition, 
           const Dtype* x, const Dtype* y, Dtype* out) {
    auto functor = WhereKernel<Dtype>();
    functor(n, condition, x, y, out);
}

template<typename Dtype>
class WhereOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    WhereOp(const std::shared_ptr<BaseTensor<float>>& condition, OpType op_type): 
            _condition(condition), GenericOp<Dtype>(op_type) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        /* 
            inputs[0] ==> x 
            inputs[1] ==> y 
         */
        assert(inputs.size()==2 && "input number of WhereOp must be 2");

        cached_data_type cached_data = __create_cached_data(inputs[1]->shape(),
                                                            inputs[1]->dtype,
                                                            inputs[1]->device());
        _n = cached_data->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in WhereOp failed");

        ApplyWhere<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  _condition->cached_ptr(),
                                                  inputs[0]->cached_ptr(),
                                                  inputs[1]->cached_ptr(),
                                                  cached_data->cached_ptr());

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyLog failed");

        cached_data->cached = true;
        cached_data->is_compact = true;

        return cached_data;
    }

    /* TODO */
    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {
        return {out_grad};
    }

protected:
    virtual inline cudaError_t _get_num_blocks() override {
        int dev, sm_count, tpm;
        cudaError err = __get_gpu_info(&dev, &sm_count, &tpm);
        //_num_blocks = std::max<int>(1, std::min<int64_t>((_n + kBlockSize - 1) / kBlockSize,
        //                                      sm_count * tpm / kBlockSize * NUMWAVES));
        _num_blocks = (_n+kBlockSize-1)/kBlockSize;
        return cudaSuccess;
    }

private:
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
    const std::shared_ptr<BaseTensor<float>> _condition;
};

#endif

