#ifndef __MAX_OP__
#define __MAX_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/permute.cuh"
#include "ops/bp/broadcast.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class ReducedMax {
public:
    __device__ void operator()(size_t n,
                               size_t reduce_size,
                               const Dtype* a, 
                               Dtype* out,
                               float* idx_ptr) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >= n) return;

        size_t offset = tx*reduce_size;
        Dtype tmp = a[offset];
        float idx = 0;

        for(size_t i=0; i<reduce_size; ++i) {
            if(tmp < a[offset+i]) {
                tmp = a[offset+i];
                idx = static_cast<float>(i);
            }
        }

        out[tx] = tmp;
        idx_ptr[tx] = idx; 
    }
};

template<typename Dtype>
class MaxGradSetitem {
public:
    __device__ void operator()(size_t n,
                               const float* idx_ptr, 
                               CudaVec shape,
                               CudaVec strides,
                               CudaVec idx_strides,
                               size_t offset,
                               int dim,
                               const Dtype* a, 
                               Dtype* out) {

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        size_t indices[MAX_VEC_SIZE];
        get_index(tid, indices, shape);

        size_t out_idx = offset;
        size_t set_idx  = offset;
        size_t idx_idx = 0;
        int cnt = 0;

        for(int i=0; i<shape.size; ++i) {
            out_idx += indices[i]*strides.data[i];
            if(i!=dim) { 
                idx_idx += indices[i]*idx_strides.data[cnt];
                set_idx += indices[i]*strides.data[i];
                cnt++;
            }
        }

        int ax_idx = idx_ptr[idx_idx];
        set_idx += ax_idx*strides.data[dim];

        if(tid<n && set_idx==out_idx)
            out[out_idx] = a[idx_idx];
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyRedMax(size_t n, size_t reduce_size, const Dtype* a, Dtype* out, float* idx_ptr) {
    auto functor = ReducedMax<Dtype>();
    functor(n, reduce_size, a, out, idx_ptr);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyMaxGrad(size_t n, const float* idx_ptr, CudaVec shape, 
             CudaVec stride, CudaVec idx_stride, 
             size_t offset, int dim, const Dtype* a, Dtype* out) {
    auto functor = MaxGradSetitem<Dtype>();
    functor(n, idx_ptr, shape, stride, idx_stride, offset, dim, a, out);
}

template<typename Dtype>
class MaxOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    using cached_fp32_type = std::shared_ptr<BaseTensor<float>>;

public:
    MaxOp(OpType op_type, int dim, bool keepdim,
          cached_fp32_type& idx_ptr): 
        GenericOp<Dtype>(op_type), _axes({dim}), 
        _keepdim(keepdim), _idx_ptr(idx_ptr), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of MaxOp must be 1");

        __prepare_pos_axes(inputs[0]->shape());

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
        assert(err==cudaSuccess && "get_num_blocks in MaxOp failed");

        ApplyRedMax<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n, final_shape,
                                                  reshape_cache->cached_ptr(),
                                                  cached_data->cached_ptr(),
                                                  _idx_ptr->cached_ptr());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyRedMax failed");

        if(_keepdim) {
            reshape_shape.insert(reshape_shape.begin()+_axes[0], 1);
            std::shared_ptr<GenericOp<Dtype>> reshape_op =
                std::make_shared<ReshapeOp<Dtype>>(reshape_shape, OpType::Reshape);
            cached_data = reshape_op->compute({cached_data});
            //reshape_cache->compact(); 
        }

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
        out_cached->zeros();

        _n = out_cached->size();
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in MaxOp failed");

        ApplyMaxGrad<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n, _idx_ptr->cached_ptr(),
                                                  VecToCuda(out_cached->shape()),
                                                  VecToCuda(out_cached->strides()),
                                                  VecToCuda(_idx_ptr->strides()),
                                                  out_cached->offset(),
                                                  _axes[0],
                                                  out_grad->cached_ptr(),
                                                  out_cached->cached_ptr());

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

    std::vector<int> _left_axes;

    cached_fp32_type _idx_ptr;
    bool _keepdim;
    std::vector<int32_t> _axes;
    std::vector<int32_t> _left_shape;
    std::vector<int32_t> _reduced_shape;
};

#endif

