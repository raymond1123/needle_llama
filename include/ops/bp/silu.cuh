#ifndef __SILU_OP__
#define __SILU_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class SiluKernel {
public:
    __device__ void operator()(const size_t n,
                               const Dtype* a, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;
        Dtype sigmoid = 1.0 / (1.0 + expf(-a[tx]));
        out[tx] = a[tx] * sigmoid;
    }
};

/* safe softmax for fp16 */
template<>
class SiluKernel<__half> {
public:
    __device__ void operator()(const size_t n,
                               const __half* a, 
                               __half* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;
        //out[tx] = a[tx]*(1+1/hexp(-a[tx]));
        __half one = __float2half(1.0f);           
        __half neg_a = __hneg(a[tx]);              // -a[tx]
        __half exp_neg_a = hexp(neg_a);            // exp(-a[tx])
        __half denominator = __hadd(one, exp_neg_a); // 1 + hexp(-a[tx])
        __half sigmoid_a = hrcp(denominator); // 1 / (1+hexp(-a[tx]))
        out[tx] = __hmul(a[tx], sigmoid_a);        // 计算 a[tx] * sigmoid(a[tx])
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplySilu(const size_t n, const Dtype* a, Dtype* out) {
    auto functor = SiluKernel<Dtype>();
    functor(n, a, out);
}

template<typename Dtype>
class SiluOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    SiluOp(OpType op_type): GenericOp<Dtype>(op_type), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of softmax must be 1");

        std::vector<int32_t> input_shape = inputs[0]->shape();
        cached_data_type cached_data = __create_cached_data(input_shape,
                                                            DataType::FLOAT, 
                                                            inputs[0]->device());
        _n = cached_data->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");
        ApplySilu<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                         inputs[0]->cached_ptr(),
                                                         cached_data->cached_ptr());

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyRMSNorm failed");

        cached_data->cached = true;
        cached_data->is_compact = true;

        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {
        /* not done yet */
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
};

#endif
