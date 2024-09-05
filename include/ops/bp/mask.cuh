#ifndef __MASK_OP__
#define __MASK_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class MaskEqTensorKernel{
public:
    __device__ void operator()(size_t n,
                               const Dtype* x, 
                               const Dtype* y, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        if(x[tx]==y[tx]) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<>
class MaskEqTensorKernel<__half>{
public:
    __device__ void operator()(size_t n,
                               const __half* x, 
                               const __half* y, 
                               __half* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        if(__heq(x[tx],y[tx])) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<typename Dtype>
class MaskNEqTensorKernel{
public:
    __device__ void operator()(size_t n,
                               const Dtype* x, 
                               const Dtype* y, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        if(x[tx]!=y[tx]) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<>
class MaskNEqTensorKernel<__half>{
public:
    __device__ void operator()(size_t n,
                               const __half* x, 
                               const __half* y, 
                               __half* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        if(__hne(x[tx], y[tx])) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<typename Dtype>
class MaskEqScalarKernel{
public:
    __device__ void operator()(size_t n,
                               const Dtype* x, 
                               const float y, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        if(x[tx]==y) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<>
class MaskEqScalarKernel<__half>{
public:
    __device__ void operator()(size_t n,
                               const __half* x, 
                               const float y, 
                               __half* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        float tmp = __half2float(x[tx]);
        if(tmp==y) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<typename Dtype>
class MaskNEqScalarKernel{
public:
    __device__ void operator()(size_t n,
                               const Dtype* x, 
                               const float y, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        if(x[tx]!=y) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<>
class MaskNEqScalarKernel<__half>{
public:
    __device__ void operator()(size_t n,
                               const __half* x, 
                               const float y, 
                               __half* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx >= n) return; 

        float tmp = __half2float(x[tx]);
        if(tmp!=y) out[tx] = 1.0;
        else out[tx] = 0.0;
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyEqMaskTensor(size_t n, const Dtype* x, const Dtype* y, Dtype* out) {
    auto functor = MaskEqTensorKernel<Dtype>();
    functor(n, x, y, out);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyNEqMaskTensor(size_t n, const Dtype* x, const Dtype* y, Dtype* out) {
    auto functor = MaskNEqTensorKernel<Dtype>();
    functor(n, x, y, out);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyEqMaskScalar(size_t n, const Dtype* x, const float y, Dtype* out) {
    auto functor = MaskEqScalarKernel<Dtype>();
    functor(n, x, y, out);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyNEqMaskScalar(size_t n, const Dtype* x, const float y, Dtype* out) {
    auto functor = MaskNEqScalarKernel<Dtype>();
    functor(n, x, y, out);
}

template<typename Dtype>
class MaskOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    using cached_fp32_type = std::shared_ptr<BaseTensor<float>>;

public:
    MaskOp(bool eq, bool cmp_scalar, OpType op_type, const float scalar=0.0): 
            GenericOp<Dtype>(op_type), _cmp_scalar(cmp_scalar), 
            _eq(eq), _scalar(scalar) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        if(_cmp_scalar)
            assert(inputs.size()==1 && "input number of MaskOp must be 1");
        else
            assert(inputs.size()==2 && "input number of MaskOp must be 2");

        cached_data_type cached_data = __create_cached_data(inputs[0]->shape(),
                                                            inputs[0]->dtype,
                                                            inputs[0]->device());
        _n = cached_data->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in MaskOp failed");

        if(_cmp_scalar){
            if(_eq)
                ApplyEqMaskScalar<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                             inputs[0]->cached_ptr(),
                                                             _scalar,
                                                             cached_data->cached_ptr());
            else
                ApplyNEqMaskScalar<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                             inputs[0]->cached_ptr(),
                                                             _scalar,
                                                             cached_data->cached_ptr());

        } else {
            if(_eq)
                ApplyEqMaskTensor<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                             inputs[0]->cached_ptr(),
                                                             inputs[1]->cached_ptr(),
                                                             cached_data->cached_ptr());
            else
                ApplyNEqMaskTensor<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                             inputs[0]->cached_ptr(),
                                                             inputs[1]->cached_ptr(),
                                                             cached_data->cached_ptr());
        }

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyMask failed");

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
    bool _eq;
    bool _cmp_scalar;

    const float _scalar;
};

#endif

