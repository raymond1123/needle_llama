#ifndef __ROTARY_OP__
#define __ROTARY_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/permute.cuh"
#include "ops/bp/broadcast.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class RotaryKernel {
public:
    __device__ void operator()(size_t n,
                               const int num_head,
                               const int head_dim,
                               const int start_pos,
                               const Dtype* a, 
                               Dtype* out) {

        int row = blockIdx.x;
        int col = threadIdx.x;

        int idx = row * blockDim.x + col;
    
        if(idx>=n) return;

        int i = col/2;
        float theta_i = float(start_pos) * powf(10000, -2*i/float(head_dim));

        float cos = cosf(theta_i);
        float sin = sinf(theta_i);

        // shuffle (x1 <-> x2, x3<->x4, ...) 
        float val = a[idx];
        float neighbor = __shfl_xor_sync(0xFFFFFFFF, val, 1);

        if(idx%2==0)
            out[idx] = val*cos - neighbor*sin; 
        else 
            out[idx] = val*cos + neighbor*sin; 
    }
};


template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyRotary(size_t n, const int num_head, const int head_dim, 
            const int start_pos, const Dtype* a, Dtype* out) {
    auto functor = RotaryKernel<Dtype>();
    functor(n, num_head, head_dim, start_pos, a, out);
}

template<typename Dtype>
class RotaryEmbOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    RotaryEmbOp(int start_pos, OpType op_type): 
            GenericOp<Dtype>(op_type), _start_pos(start_pos), _num_blocks(0) {}

    /* only can handle shape=[1,1,num_head, head_dim]*/
    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        std::vector<int32_t> input_shape = inputs[0]->shape();
        int shape_len = input_shape.size();
        _num_head = 1;
        for(int i=0; i<shape_len-1; ++i)
            _num_head *= input_shape[i];
        _head_dim = input_shape[shape_len-1];

        cached_data_type cached_data = __create_cached_data(input_shape,
                                                            DataType::FLOAT, 
                                                            inputs[0]->device());
        _n = cached_data->size();
        //cudaError_t err = _get_num_blocks();
        //assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");

        ApplyRotary<Dtype><<<_num_head, _head_dim, 0>>>(_n, _num_head, 
                                                        _head_dim,
                                                        _start_pos,
                                                        inputs[0]->cached_ptr(),
                                                        cached_data->cached_ptr());

        cudaError err = cudaPeekAtLastError();
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

    int _start_pos;
    int _num_head;
    int _head_dim;
};

#endif

