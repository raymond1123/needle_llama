#ifndef __RMSNORM_OP__
#define __RMSNORM_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/permute.cuh"
#include "ops/bp/broadcast.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class NormKernel {
public:
    __device__ void operator()(size_t n,
                               size_t reduce_size,
                               Dtype eps, 
                               const Dtype* a, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        size_t offset = tx*reduce_size;
        Dtype tmp = 0;

        for(size_t i=0; i<reduce_size; ++i) {
            tmp += powf(a[offset+i], 2);
        }

        tmp = rsqrtf(tmp / static_cast<Dtype>(reduce_size) + eps);

        for(size_t i=0; i<reduce_size; ++i) {
            out[offset+i] = a[offset+i] * tmp;
        }
    }
};

template<>
class NormKernel<__half> {
public:
    __device__ void operator()(size_t n,
                               int reduce_size,
                               __half eps,
                               const __half* a, 
                               __half* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        size_t offset = tx*reduce_size;
        __half tmp = 0;

        for(size_t i=0; i<reduce_size; ++i) {
            tmp = __hadd(tmp, __hmul(a[offset+i], a[offset+i]));
            tmp = __hadd(tmp, eps); 
        }

        tmp = __hdiv(tmp, static_cast<__half>(reduce_size)); 
        tmp = hrsqrt(__hadd(eps, tmp));

        for(size_t i=0; i<reduce_size; ++i) {
            out[offset+i] = __hmul(tmp, a[offset+i]);
        }
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyNorm(size_t n, size_t reduce_size, const Dtype* a, Dtype* out, Dtype eps=1e-6) {
    auto functor = NormKernel<Dtype>();
    functor(n, reduce_size, eps, a, out);
}

template<typename Dtype>
class RMSNormOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    RMSNormOp(OpType op_type): GenericOp<Dtype>(op_type), 
                               _axes({-1}), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of RMSNorm must be 1");

        inputs[0]->compact();
        __prepare_pos_axes(inputs[0]->shape());

        cached_data_type cached_data = __create_cached_data(inputs[0]->shape(),
                                                            inputs[0]->dtype,
                                                            inputs[0]->device());
        _n = 1;
        for(int i=0; i<inputs[0]->shape().size()-1; ++i)
            _n *= inputs[0]->shape()[i];

        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");

        int final_shape = (inputs[0]->shape())[_axes[0]];
        ApplyNorm<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n, final_shape,
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
        auto inputs = tensor->inputs;
        std::vector<int32_t> input_shape = inputs[0]->shape();
        std::vector<int32_t> reshape_shape = input_shape;

        for(auto& axis: _axes)
            reshape_shape[axis] = 1;

        std::shared_ptr<GenericOp<Dtype>> reshape_op =
            std::make_shared<ReshapeOp<Dtype>>(reshape_shape, OpType::Reshape);
        cached_data_type reshape_cache = reshape_op->compute({out_grad});

        std::shared_ptr<GenericOp<Dtype>> broadcast_op = 
            std::make_shared<BroadcastOp<Dtype>>(input_shape, OpType::BroadcastTo);
        cached_data_type out_cache = broadcast_op->compute({reshape_cache});
        out_cache->compact(); // compact here before reduced add

        return {out_cache};
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
    inline void __prepare_pos_axes(std::vector<int32_t> input_shape) {

        int length_shape = input_shape.size();
        std::vector<int> pos_axes = _axes;

        for(int i=0; i<pos_axes.size(); ++i) 
            if(pos_axes[i]<0) pos_axes[i] += length_shape;
        std::sort(pos_axes.begin(), pos_axes.end());

        for(int i=0; i<length_shape; ++i) {
            bool in_axes = false;

            for(int j=0; j<pos_axes.size(); ++j) {
                if(i==pos_axes[j]) {
                    in_axes = true;
                    _reduced_shape.push_back(input_shape[i]);
                    break;
                }
            }

            if(!in_axes) {
                _left_axes.push_back(i);
                _left_shape.push_back(input_shape[i]);
            }
        }

        _axes = pos_axes;
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

    std::vector<int> _axes;
    std::vector<int32_t> _left_axes;
    std::vector<int32_t> _left_shape;
    std::vector<int32_t> _reduced_shape;
};

#endif

