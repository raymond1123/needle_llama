#ifndef __EXP_OP__
#define __EXP_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class Exp {
public:
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx < n) out[tx] = __expf(a[tx]);
    }
};

template<>
class Exp<float> {
public:
    __device__ void operator()(size_t n, const float* a, float* out) {
        size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tx < n) {
            out[tx] = __expf(a[tx]);  // 对 float 类型使用 logf 函数
        }
    }
};

// Dtype ==> __half
template<>
class Exp<__half> {
public:
    __device__ void operator()(size_t n, const __half* a, __half* out) {
        size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tx < n) {
            out[tx] = hexp(a[tx]);  // 对 __half 类型使用 __hlog 函数
        }
    }
};

template<typename Dtype>
class ExpGrad {
public:
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               const Dtype* b, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx < n) out[tx] = b[tx]*exp(a[tx]);
    }
};

// Dtype ==> float 
template<>
class ExpGrad<float> {
public:
    __device__ void operator()(size_t n, 
                               const float * a, 
                               const float * b, 
                               float * out) {
        size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tx < n) out[tx] = b[tx]*__expf(a[tx]);
    }
};

// Dtype ==> __half
template<>
class ExpGrad<__half> {
public:
    __device__ void operator()(size_t n, 
                               const __half* a, 
                               const __half* b, 
                               __half* out) {
        size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tx < n) out[tx] = __hmul(b[tx], hexp(a[tx]));
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyExp(size_t n, const Dtype* a, Dtype* out) {
    auto functor = Exp<Dtype>();
    functor(n, a, out);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyGradExp(size_t n, const Dtype* a, const Dtype* b, Dtype* out) {
    auto functor = ExpGrad<Dtype>();
    functor(n, a, b, out);
}

template<typename Dtype>
class ExpOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    ExpOp(OpType op_type): GenericOp<Dtype>(op_type) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of ExpOp must be 1");

        cached_data_type cached_data = __create_cached_data(inputs[0]->shape(),
                                                            inputs[0]->dtype,
                                                            inputs[0]->device());
        _n = cached_data->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in ExpOp failed");

        ApplyExp<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  inputs[0]->cached_ptr(),
                                                  cached_data->cached_ptr());

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyExp failed");

        cached_data->cached = true;
        cached_data->is_compact = true;

        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {
        auto inputs = tensor->inputs;
        std::vector<int32_t> input_shape = inputs[0]->shape();

        cached_data_type out_cached = __create_cached_data(input_shape,
                                                     out_grad->dtype,
                                                     out_grad->device());
        _n = out_cached->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in ExpOp failed");

        ApplyGradExp<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  inputs[0]->cached_ptr(),
                                                  out_grad->cached_ptr(),
                                                  out_cached->cached_ptr());

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyExpGrad failed");

        return {out_cached};
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
    cached_data_type _idx_ptr;
};

#endif

