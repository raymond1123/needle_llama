#ifndef __OPS_UTIL__
#define __OPS_UTIL__

#include "backend/cuda_util.cuh"

constexpr int kBlockSize = 256;
#define NUMWAVES 32

template<typename Dtype>
class EwSetitem {
public:
    __device__ void operator()(const Dtype* a, 
                               Dtype* out,
                               size_t size,
                               CudaVec shape,
                               CudaVec strides,
                               size_t offset) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        size_t indices[MAX_VEC_SIZE];
        get_index(tid, indices, shape);

        size_t out_idx = offset;
        for (int i=0; i<shape.size; ++i)
          out_idx += indices[i]*strides.data[i];

        if(tid<size)
          out[out_idx] = a[tid];
    }
};

template<typename Dtype>
class EwSetscalar {
public:
    __device__ void operator()(const Dtype a, 
                               Dtype* out,
                               size_t size,
                               CudaVec shape,
                               CudaVec strides,
                               size_t offset) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        size_t indices[MAX_VEC_SIZE];
        get_index(tid, indices, shape);

        size_t out_idx = offset;
        for (int i=0; i<shape.size; ++i)
          out_idx += indices[i]*strides.data[i];

        if(tid<size)
          out[out_idx] = a;
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyEwSetitem(size_t n, const Dtype* a, Dtype* out, 
           CudaVec shape, CudaVec strides, size_t offset) {
    auto functor = EwSetitem<Dtype>();
    functor(a, out, n, shape, strides, offset);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyEwSetscalar(size_t n, const Dtype a, Dtype* out, 
           CudaVec shape, CudaVec strides, size_t offset) {
    auto functor = EwSetscalar<Dtype>();
    functor(a, out, n, shape, strides, offset);
}

#endif

