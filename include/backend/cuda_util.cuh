#ifndef __CUDA_UTIL_HPP__
#define __CUDA_UTIL_HPP__

#define MAX_VEC_SIZE 8
#define BASE_THREAD_NUM 256

struct CudaVec {
  size_t size;
  int32_t data[MAX_VEC_SIZE];
};

struct CudaDims {
  dim3 block, grid;
};

static CudaVec VecToCuda(const std::vector<int32_t>& x) {
    CudaVec shape;
    if (x.size() > MAX_VEC_SIZE) 
        throw std::runtime_error("Exceeded CUDA supported max dimesions");

    shape.size = x.size();
    for (size_t i = 0; i < x.size(); i++) {
      shape.data[i] = x[i];
    }
    return shape;
}

static CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

__device__ void get_index(size_t gid, size_t *indices, CudaVec &shape) {
    uint32_t cur=1, pre=1;
    for(int i=shape.size-1; i>=0; --i) {
        cur *= shape.data[i];
        indices[i] = gid%cur/pre;
        pre *= shape.data[i];
    }
}

#endif
