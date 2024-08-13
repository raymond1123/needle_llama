#ifndef __LOGSUMEXP_OP__
#define __LOGSUMEXP_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/summation.cuh"
#include "ops/bp/log.cuh"
#include "ops/bp/exp.cuh"
#include "ops/ops_util.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class ExpMinux {
public:
    /* out = exp(a - b) */
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               const Dtype* b, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        out[tx] = exp(a[tx]-b[tx]);
    }
};

template<typename Dtype>
class LogSum {
public:
    /* out = log(a) + b */
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               const Dtype* b, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        out[tx] = log(a[tx])+b[tx];
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyExpMinux(size_t n, const Dtype* a, const Dtype* b, Dtype* out) {
    auto functor = ExpMinux<Dtype>();
    functor(n, a, b, out);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyLogSum(size_t n, const Dtype* a, const Dtype* b, Dtype* out) {
    auto functor = LogSum<Dtype>();
    functor(n, a, b, out);
}

template<typename Dtype>
class LogSumExpOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    LogSumExpOp(int dim, OpType op_type):
        GenericOp<Dtype>(op_type), _axes({dim}), _num_blocks(0) {

            auto input_shape = inputs[0]->shape();
            if(_axes[0]<0) _axes[0] += input_shape.size();

        }

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of LogSumExpOp must be 1");
        auto input_shape = inputs[0]->shape();

        std::shared_ptr<GenericOp<Dtype>> max_op = 
            std::make_shared<MaxOp<Dtype>>(OpType::Max, _axes[0], true);
        cached_data_type max_z_original = max_op->compute({inputs[0]});

        std::shared_ptr<GenericOp<Dtype>> max_op = 
            std::make_shared<MaxOp<Dtype>>(OpType::Max, _axes[0]);
        cached_data_type max_z_reduce = max_op->compute({inputs[0]});

        std::shared_ptr<GenericOp<Dtype>> broadcast_op = 
            std::make_shared<BroadcastOp<Dtype>>(inputs[0]->shape(), 
                                                 OpType::BroadcastTo);
        cached_data_type broadcast_t = broadcast_op->compute({max_z_original});

        cached_data_type exp_t = __create_cached_data(input_shape,
                                                      inputs[0]->device());
        _n = exp_t->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in MaxOp failed");

        /* exp(input - max_z_original) */
        ApplyExpMinux<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  inputs[0]->cached_ptr(),
                                                  broadcast_t->cached_ptr(),
                                                  exp_t->cached_ptr());

        std::shared_ptr<GenericOp<Dtype>> sum_op = 
            std::make_shared<ExpOp<Dtype>>(_axes, OpType::Summation);
        cached_data_type exp_sum = sum_op->compute({exp_t});

        cached_data_type out = __create_cached_data(exp_sum->shape(),
                                                    inputs[0]->device());
        _n = out->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in MaxOp failed");

        /* log(exp_sum) + max_z_reduce */
        /* out = log(a) + b */
        ApplyLogSum<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  exp_sum->cached_ptr(),
                                                  max_z_reduce->cached_ptr(),
                                                  out->cached_ptr());

        return out;
    }

    // TODO
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
    inline void __prepare_zero_axes(std::vector<int32_t> input_shape) {
        for(int i=0; i<input_shape.size(); ++i)
            _axes.push_back(i);

        _reduced_shape = input_shape;
        _left_shape = {1};
    }

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
                                                 BackendType device,
                                                 bool create_cache=true) {
        cached_data_type cached_data = nullptr;
        if (device == BackendType::CPU) {
            cached_data.reset(new CpuTensor<Dtype>(shape, create_cache));
        } else if (device == BackendType::CUDA) {
            cached_data.reset(new CudaTensor<Dtype>(shape, create_cache));
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }

        return cached_data;
    }

private:
    size_t _n;
    int _num_blocks;

    std::vector<int> _axes;
    std::vector<int> _left_axes;

    std::vector<int32_t> _left_shape;
    std::vector<int32_t> _reduced_shape;
};

#endif

