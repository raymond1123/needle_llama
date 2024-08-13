#ifndef __RELU_OP__
#define __RELU_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class Relu {
public:
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               Dtype* out,
                               Dtype* idx) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        if(a[tx] < static_cast<Dtype>(0)) {
            out[tx] = static_cast<Dtype>(0);
            idx[tx] = static_cast<Dtype>(0);
        } else {
            out[tx] = a[tx];
            idx[tx] = static_cast<Dtype>(1);
        }
    }
};

template<typename Dtype>
class ReluGrad {
public:
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               Dtype* out,
                               const Dtype* idx) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        if(idx[tx] == static_cast<Dtype>(0)) {
            out[tx] = static_cast<Dtype>(0);
        } else {
            out[tx] = a[tx];
        }
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyRelu(size_t n, const Dtype* a, Dtype* out, Dtype* idx_ptr) {
    auto functor = Relu<Dtype>();
    functor(n, a, out, idx_ptr);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyReluGrad(size_t n, const Dtype* a, Dtype* out, const Dtype* idx_ptr) {
    auto functor = ReluGrad<Dtype>();
    functor(n, a, out, idx_ptr);
}

template<typename Dtype>
class ReluOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    ReluOp(OpType op_type): GenericOp<Dtype>(op_type) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of ReluOp must be 1");

        cached_data_type cached_data = __create_cached_data(inputs[0]->shape(),
                                                            inputs[0]->dtype,
                                                            inputs[0]->device());

        _idx_ptr = __create_cached_data(inputs[0]->shape(), 
                                        inputs[0]->dtype,
                                        inputs[0]->device());
        _idx_ptr->zeros();

        _n = cached_data->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in ReluOp failed");

        ApplyRelu<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  inputs[0]->cached_ptr(),
                                                  cached_data->cached_ptr(),
                                                  _idx_ptr->cached_ptr());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyRelu failed");

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
        assert(err==cudaSuccess && "get_num_blocks in ReluOp failed");

        ApplyReluGrad<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n, 
                                                             out_grad->cached_ptr(),
                                                             out_cached->cached_ptr(),
                                                             _idx_ptr->cached_ptr());
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

