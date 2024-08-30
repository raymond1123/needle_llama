#ifndef __EMBD_OP__
#define __EMBD_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/permute.cuh"
#include "ops/bp/broadcast.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class EmbeddingKernel {
public:
    __device__ void operator()(const int n,
                               const int dim,
                               const Dtype* emb, 
                               const float* idx, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        size_t offset = int(idx[blockIdx.x])*dim;
        out[tx] = emb[threadIdx.x+offset];
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyEmbedding(const size_t n, const int dim, 
               const Dtype* emb, const float* idx, Dtype* out) {
    auto functor = EmbeddingKernel<Dtype>();
    functor(n, dim, emb, idx, out);
}

template<typename Dtype>
class EmbeddingOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    EmbeddingOp(const std::shared_ptr<BaseTensor<float>>& index, OpType op_type): 
            GenericOp<Dtype>(op_type), _num_blocks(0), _index(index) {}

    /* only can handle shape=[1, 1, num_head, head_dim]*/
    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of embedding must be 1");

        int32_t vocab_size = inputs[0]->shape()[0];
        int32_t dim = inputs[0]->shape()[1];
        int32_t batch_size = _index->shape()[0];
        int32_t seq_len = _index->shape()[1];

        std::vector<int32_t> out_shape = {batch_size, seq_len, dim};
        cached_data_type cached_data = __create_cached_data(out_shape,
                                                            inputs[0]->dtype,
                                                            inputs[0]->device());
        //_n = cached_data->size();
        //cudaError_t err = _get_num_blocks();
        //assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");
        _num_blocks = batch_size*seq_len;
        ApplyEmbedding<Dtype><<<_num_blocks, dim, 0>>>(cached_data->size(), dim,
                                                       inputs[0]->cached_ptr(),
                                                       _index->cached_ptr(),
                                                       cached_data->cached_ptr());

        cudaError_t err = cudaPeekAtLastError();
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
    const std::shared_ptr<BaseTensor<float>> _index;
};

#endif
