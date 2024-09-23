#ifndef __CUDA_BACKEND_HPP__
#define __CUDA_BACKEND_HPP__

#include "common.hpp"
#include "backend/cuda_util.cuh"

template<typename Dtype>
__global__ void CompactKernel(const Dtype* a, Dtype* out, 
                              size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t indices[MAX_VEC_SIZE];
    get_index(tid, indices, shape);

    size_t in_idx = offset;
    for (int i=0; i<shape.size; ++i)
        in_idx += indices[i]*strides.data[i];

    if (tid<size) {
        out[tid] = a[in_idx];
    }
}

template<typename Dtype>
__global__ void convert_fp32_to_fp16(Dtype* out, const float* data, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < size) {
        out[gid] = __float2half(data[gid]);
    }
}

template<typename Dtype>
__global__ void convert_fp16_to_fp32(Dtype* out, const __half* data, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < size) {
        out[gid] = __half2float(data[gid]);
    }
}

template<typename Dtype>
__global__ void Arange(Dtype* out, Dtype start,
                       Dtype step, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid==0) out[gid] = start;
  else if (gid < size) {
      out[gid] = start+gid*step;
  }
}

template<>
__global__ void Arange(__half* out, __half start, __half step, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid == 0) {
        out[gid] = start;
    } else if (gid < size) {
        __half gid_half = __float2half(static_cast<float>(gid));
        out[gid] = __hadd(start, __hmul(__half(gid_half), step));
    }
}

template<typename Dtype>
__global__ void FillKernel(Dtype* out, Dtype val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

template<typename Dtype>
class CudaArray:public BaseArray<Dtype> {
public:
    using cached_array_type = std::shared_ptr<BaseArray<Dtype>>;

    CudaArray(const size_t size, bool cached_data=true);
    ~CudaArray() {
        cudaFree(this->__ptr);
        this->__size = 0;
    }

    CudaArray(const CudaArray&)=delete;
    CudaArray& operator=(const CudaArray&)=delete;
    //CudaArray& operator=(const CudaArray&);
    virtual void half(const float* data) override;
    virtual void to_float(const __half* data) override;

    virtual void mem_cpy(Dtype* ptr, 
                         MemCpyType mem_cpy_type) override;

    virtual void fill_val(Dtype val) override;
    virtual void arange(Dtype start, Dtype step);
    virtual cached_array_type compact(size_t size, 
                              std::vector<int32_t> shape,
                              std::vector<int32_t> strides,
                              size_t offset) override;

    void half2numpy(float* data) override;

private:
    cudaError_t __device2host(Dtype* host_ptr);
    cudaError_t __host2device(const Dtype* host_ptr);
    cudaError_t __device2device(Dtype* other_device_ptr);
};

template<typename Dtype>
CudaArray<Dtype>::CudaArray(const size_t size, bool create_cache): 
    BaseArray<Dtype>(size) {

    if(create_cache) {
        cudaError_t err = cudaMalloc(&this->__ptr, this->__size*sizeof(Dtype));
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
}

/*
template<typename Dtype>
CudaArray<Dtype>& CudaArray<Dtype>::operator=(const CudaArray<Dtype>& other) { 
    if(this==&other) return *this;

    //BaseArray<Dtype>(size);

}
*/

template<typename Dtype>
void CudaArray<Dtype>::mem_cpy(Dtype* ptr, 
                               MemCpyType mem_cpy_type) {
    cudaError_t err = cudaSuccess;
    if (mem_cpy_type==MemCpyType::Host2Dev) {
        err = __host2device(ptr);
    } else if(mem_cpy_type==MemCpyType::Dev2Host) {
        err = __device2host(ptr);
    } else if(mem_cpy_type==MemCpyType::Dev2Dev) {
        err = __device2device(ptr);
    }

    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template<typename Dtype>
cudaError_t CudaArray<Dtype>::__device2host(Dtype* host_ptr) {
    cudaError_t err = cudaMemcpy(host_ptr, this->__ptr, 
                                 this->__size * sizeof(Dtype), 
                                 cudaMemcpyDeviceToHost);
    return err;
}

template<typename Dtype>
cudaError_t CudaArray<Dtype>::__host2device(const Dtype* host_ptr) {
    cudaError_t err = cudaMemcpy(this->__ptr, host_ptr,
                                 this->__size * sizeof(Dtype), 
                                 cudaMemcpyHostToDevice);
    return err;
}

template<typename Dtype>
cudaError_t CudaArray<Dtype>::__device2device(Dtype* other_device_ptr) {
    cudaError_t err = cudaMemcpy(other_device_ptr, this->__ptr, 
                                 this->__size * sizeof(Dtype), 
                                 cudaMemcpyDeviceToDevice);
    return err;
}

template<typename Dtype>
void CudaArray<Dtype>::arange(Dtype start, Dtype step) {
    int block = 256;
    int grid = (this->__size+block-1)/block;
    Arange<<<grid, block>>>(this->__ptr, start, step, this->__size);
}

template<typename Dtype>
void CudaArray<Dtype>::half(const float* data) {
    int block = 256;
    int grid = (this->__size+block-1)/block;
    if(data != nullptr) {
        convert_fp32_to_fp16<<<grid, block>>>(this->__ptr, data, this->__size);
    }
}

template<typename Dtype>
void CudaArray<Dtype>::to_float(const __half* data) {
    int block = 256;
    int grid = (this->__size+block-1)/block;
    if(data != nullptr) {
        convert_fp16_to_fp32<<<grid, block>>>(this->__ptr, data, this->__size);
    }
}

template<typename Dtype>
void CudaArray<Dtype>::half2numpy(float* data) {
    if constexpr (std::is_same_v<Dtype, __half>) {
        int block = 256;
        int grid = (this->__size+block-1)/block;
        if(data != nullptr) {
            convert_fp16_to_fp32<<<grid, block>>>(data, this->__ptr, this->__size);
        }
    }
}

template<typename Dtype>
void CudaArray<Dtype>::fill_val(Dtype val) {
    /*
    if(val==static_cast<Dtype>(0)) {
        cudaError_t err = cudaMemset(this->__ptr, val, 
                                     this->__size*sizeof(Dtype));
        if (err != cudaSuccess) 
            throw std::runtime_error(cudaGetErrorString(err));
    } else {
        int block = 256;
        int grid = (this->__size+block-1)/block;
        FillKernel<<<grid, block>>>(this->__ptr, val, this->__size);
    }
    */
    int block = 256;
    int grid = (this->__size+block-1)/block;
    FillKernel<<<grid, block>>>(this->__ptr, val, this->__size);
}

template<typename Dtype>
std::shared_ptr<BaseArray<Dtype>> CudaArray<Dtype>::compact(size_t size, 
                                            std::vector<int32_t> shape,
                                            std::vector<int32_t> strides,
                                            size_t offset) {

    cached_array_type new_array = std::make_shared<CudaArray<Dtype>>(size);

    CudaDims dim = CudaOneDim(size);
    CompactKernel<Dtype><<<dim.grid, dim.block>>>(this->__ptr,
                                                  new_array->get_ptr(),
                                                  size,
                                                  VecToCuda(shape),
                                                  VecToCuda(strides), 
                                                  offset);
    return new_array;
}

#endif
