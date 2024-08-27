#ifndef __SOFTMAX_OP__
#define __SOFTMAX_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/permute.cuh"
#include "ops/bp/broadcast.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class SoftmaxKernel {
public:
    __device__ void operator()(size_t n,
                               const int reduce_size,
                               const Dtype* a, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        size_t offset = tx*reduce_size;
        Dtype tmp = 0;

        for(size_t i=0; i<reduce_size; ++i) {
            tmp += expf(a[offset+i]);
        }

        for(size_t i=0; i<reduce_size; ++i) {
            out[offset+i] = expf(a[offset+i]) / tmp;
        }
    }
};

/* safe softmax for fp16 */
template<>
class SoftmaxKernel<__half> {
public:
    __device__ void operator()(size_t n,
                               const int reduce_size,
                               const __half* a, 
                               __half* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        size_t offset = tx*reduce_size;
        __half tmp = 0, safe_max = a[offset];

        for(size_t i=1; i<reduce_size; ++i) {
            __half other = a[offset+i];
            if(__hgt(other, safe_max))
                safe_max = other; 
        }

        for(size_t i=0; i<reduce_size; ++i) {
            tmp += (hexp(a[offset+i]-safe_max));
        }

        for(size_t i=0; i<reduce_size; ++i) {
            out[offset+i] = hexp(a[offset+i]-safe_max) / tmp;
        }
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplySoftmax(size_t n, const int reduce_size, const Dtype* a, Dtype* out) {
    auto functor = SoftmaxKernel<Dtype>();
    functor(n, reduce_size, a, out);
}

template<typename Dtype>
class SoftmaxOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    SoftmaxOp(OpType op_type): 
            GenericOp<Dtype>(op_type), _num_blocks(0) {}

    /* only can handle shape=[1,1,num_head, head_dim]*/
    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of softmax must be 1");

        std::vector<int32_t> input_shape = inputs[0]->shape();
        int shape_len = input_shape.size();
        _dim = input_shape[shape_len-1];

        cached_data_type cached_data = __create_cached_data(input_shape,
                                                            DataType::FLOAT, 
                                                            inputs[0]->device());
        _n = 1;
        for(int i=0; i<inputs[0]->shape().size()-1; ++i)
            _n *= inputs[0]->shape()[i];
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");
        ApplySoftmax<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n, _dim, 
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

    int _dim;
};

#endif

