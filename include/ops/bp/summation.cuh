#ifndef __SUMMATION_OP__
#define __SUMMATION_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/permute.cuh"
#include "ops/bp/broadcast.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class ReducedSum {
public:
    __device__ void operator()(size_t n,
                               size_t reduce_size,
                               const Dtype* a, 
                               Dtype* sum) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        size_t offset = tx*reduce_size;
        Dtype tmp = 0;

        for(size_t i=0; i<reduce_size; ++i) {
            tmp += a[offset+i];
        }

        sum[tx] = tmp;
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyRedSum(size_t n, size_t reduce_size, const Dtype* a, Dtype* sum) {
    auto functor = ReducedSum<Dtype>();
    functor(n, reduce_size, a, sum);
}

template<typename Dtype>
class SummationOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    SummationOp(std::vector<int> axes, OpType op_type):
        GenericOp<Dtype>(op_type), _axes(axes), _num_blocks(0) {}

    SummationOp(OpType op_type): GenericOp<Dtype>(op_type), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of SummationOp must be 1");

        if(_axes.size()==0) {
            __prepare_zero_axes(inputs[0]->shape());
        } else {
            __prepare_pos_axes(inputs[0]->shape());
        }

        std::vector<int> permute_axes(_left_axes.begin(), _left_axes.end());
        permute_axes.insert(permute_axes.end(), _axes.begin(), _axes.end());

        // first permute 
        std::shared_ptr<GenericOp<Dtype>> permute_op = 
            std::make_shared<PermuteOp<Dtype>>(permute_axes, OpType::Permute);
        cached_data_type permute_cache = permute_op->compute({inputs[0]});

        size_t final_shape = 1;
        for(auto& s: _reduced_shape)
            final_shape *= s;

        // second reshape
        std::vector<int32_t> reshape_shape = _left_shape;
        reshape_shape.push_back(final_shape);

        std::shared_ptr<GenericOp<Dtype>> reshape_op =
            std::make_shared<ReshapeOp<Dtype>>(reshape_shape, OpType::Reshape);
        cached_data_type reshape_cache = reshape_op->compute({permute_cache});
        reshape_cache->compact(); // compact here before reduced add

        cached_data_type cached_data = __create_cached_data(_left_shape,
                                                            inputs[0]->dtype,
                                                            inputs[0]->device());
        _n = cached_data->size();

        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");

        ApplyRedSum<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n, final_shape,
                                                  reshape_cache->cached_ptr(),
                                                  cached_data->cached_ptr());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyRedSum failed");

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
    std::vector<int> _left_axes;

    std::vector<int32_t> _left_shape;
    std::vector<int32_t> _reduced_shape;
};

#endif

