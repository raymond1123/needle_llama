#ifndef __MATMUL_OP__
#define __MATMUL_OP__

#include <cublas_v2.h>

#include "ops/generic_op.cuh"
#include "ops/bp/reshape.cuh"
#include "ops/bp/broadcast.cuh"
#include "ops/bp/summation.cuh"

#define BASE_THREAD_NUM 256

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class Gemm {
public:
    __device__ void operator()(const Dtype* A, 
                               const Dtype* B, 
                               Dtype* C,
                               int M, int K, int N) {

        size_t row = blockDim.x*blockIdx.x+threadIdx.x;
        size_t col = blockDim.y*blockIdx.y+threadIdx.y;

        if(row < M && col < N) {
            C[row*N+col] = 0;

            for(uint32_t i=0; i<K; ++i) {
                uint32_t a_idx = row*K+i;
                uint32_t b_idx = i*N+col;

                C[row*N+col] += A[a_idx] * B[b_idx];
            }
        }
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyGemm(const Dtype* A, const Dtype* B, Dtype* C, int M, int K, int N) {
    auto functor = Gemm<Dtype>();
    functor(A,B,C,M,K,N);
}

void ApplyGemm_fp16(std::vector<int> &first_shape, 
                    std::vector<std::shared_ptr<BaseTensor<__half>>> gemm_AB,
                    std::shared_ptr<BaseTensor<__half>> gemm_C,
                    int M, int K, int N) {

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    static half alpha = 1.0;
    static half beta = 0.0;
    int batch_count = first_shape[0];

    size_t offset_A = gemm_AB[0]->strides()[0];
    size_t offset_B = gemm_AB[1]->strides()[0];
    size_t offset_C = gemm_C->strides()[0];
    for(int i=0; i < batch_count; ++i) {

        __half* A = gemm_AB[0]->cached_ptr() + i*offset_A;
        __half* B = gemm_AB[1]->cached_ptr() + i*offset_B;
        __half* C = gemm_C->cached_ptr() + i*offset_C;

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                     M, N, K, 
                     &alpha, 
                     B, CUDA_R_16F, K, 
                     A, CUDA_R_16F, K, 
                     &beta, 
                     C, CUDA_R_16F, N, 
                     CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

void ApplyGemm_fp32(std::vector<int> &first_shape, 
                    std::vector<std::shared_ptr<BaseTensor<float>>> gemm_AB,
                    std::shared_ptr<BaseTensor<float>> gemm_C,
                    int M, int K, int N) {
    cublasHandle_t handle;
    cublasStatus_t status;
    float alpha = 1.0,beta = 0.0;

    int batch_count = first_shape[0];

    status = cublasCreate(&handle);
    size_t offset_A = gemm_AB[0]->strides()[0];
    size_t offset_B = gemm_AB[1]->strides()[0];
    size_t offset_C = gemm_C->strides()[0];
    for(int i=0; i < batch_count; ++i) {

        float* A = gemm_AB[0]->cached_ptr() + i*offset_A;
        float* B = gemm_AB[1]->cached_ptr() + i*offset_B;
        float* C = gemm_C->cached_ptr() + i*offset_C;

        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, M, K, 
                             &alpha, 
                             B, N, 
                             A, K, 
                             &beta, 
                             C, N);

        assert(status == CUBLAS_STATUS_SUCCESS && "cublasSgemmBatched failed");
    }
}

template<typename Dtype>
class MatMulOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    MatMulOp(std::vector<int> axes, OpType op_type):
        GenericOp<Dtype>(op_type),  _num_blocks(0) {}

    MatMulOp(OpType op_type): GenericOp<Dtype>(op_type), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        int input_length1=inputs[0]->shape().size();
        int input_length2=inputs[1]->shape().size();
        int max_shape_len = std::max(input_length1, input_length2);

        assert(inputs.size()==2 && "input number of MatMul must be 2");
        assert(inputs[0]->shape()[input_length1-1]==inputs[1]->shape()[input_length2-2]
               && "input number of MatMul must be 2");

        int bidx = __broadcast_axes(inputs[0]->shape(), inputs[1]->shape());

        // first need to broadcast
        if (bidx > -1) {
            std::shared_ptr<GenericOp<Dtype>> broadcast_op = 
                std::make_shared<BroadcastOp<Dtype>>(_bshape, OpType::BroadcastTo);
            inputs[bidx] = broadcast_op->compute({inputs[bidx]});
        }

        int M = inputs[0]->shape()[max_shape_len-2];
        int K = inputs[0]->shape()[max_shape_len-1];
        int N = inputs[1]->shape()[max_shape_len-1];


        // MxK @ KxN = MxN
        std::vector<cached_data_type> gemm_AB;
        std::vector<int> first_shape = {1,1};

        for(int i=0; i<2; ++i) {
            inputs[i]->compact();
            auto reshape_shape = __prepare_shape(inputs[i]->shape());
            first_shape[i] = reshape_shape[0];

            std::shared_ptr<GenericOp<Dtype>> reshape_op =
                std::make_shared<ReshapeOp<Dtype>>(reshape_shape, OpType::Reshape);
            cached_data_type reshape_cache = reshape_op->compute({inputs[i]});
            gemm_AB.push_back(reshape_cache);
        }

        assert(first_shape[0] == first_shape[1] && "gemm shape does not match");

        std::vector<int32_t> out_shape = {first_shape[0], M, N};
        std::vector<int32_t> final_shape;

        for(int i=0; i<max_shape_len-2; ++i)
            final_shape.push_back(inputs[0]->shape()[i]);
        final_shape.push_back(M);
        final_shape.push_back(N);

        cached_data_type gemm_C = __create_cached_data(out_shape, 
                                                       inputs[0]->dtype,
                                                       inputs[0]->device());
        _n = gemm_C->size();

        /*
        cudaError_t err = _get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");

        size_t m = (M + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
        size_t n = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;

        dim3 grid(BASE_THREAD_NUM, BASE_THREAD_NUM, 1);
        dim3 block(m, n, 1);

        size_t offset_A = gemm_AB[0]->strides()[0];
        size_t offset_B = gemm_AB[1]->strides()[0];
        size_t offset_C = gemm_C->strides()[0];

        for(int i=0; i < first_shape[0]; ++i) {

            Dtype* A = gemm_AB[0]->cached_ptr() + i*offset_A;
            Dtype* B = gemm_AB[1]->cached_ptr() + i*offset_B;
            Dtype* C = gemm_C->cached_ptr() + i*offset_C;

            ApplyGemm<Dtype><<<grid, block>>>(A, B, C, M, K, N);
        }

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyGemm failed");
        */

        if constexpr (std::is_same_v<Dtype, float>)
            ApplyGemm_fp32(first_shape, gemm_AB, gemm_C, M, K, N);
        if constexpr (std::is_same_v<Dtype, __half>)
            ApplyGemm_fp16(first_shape, gemm_AB, gemm_C, M, K, N);

        std::shared_ptr<GenericOp<Dtype>> reshape_op =
            std::make_shared<ReshapeOp<Dtype>>(final_shape, OpType::Reshape);
        cached_data_type out = reshape_op->compute({gemm_C});

        out->cached = true;
        out->is_compact = true;

        return out;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {
        auto inputs = tensor->inputs;
        auto a_org_shape = inputs[0]->shape();
        auto b_org_shape = inputs[1]->shape();
        auto a_org_size = inputs[0]->size();
        auto b_org_size = inputs[1]->size();

        auto a_trans_axes = __calc_trans_axes(inputs[0]->shape());
        auto b_trans_axes = __calc_trans_axes(inputs[1]->shape());

        std::shared_ptr<GenericOp<Dtype>> a_trans_op = 
            std::make_shared<TransposeOp<Dtype>>(a_trans_axes, OpType::Transpose);
        std::shared_ptr<GenericOp<Dtype>> b_trans_op = 
            std::make_shared<TransposeOp<Dtype>>(b_trans_axes, OpType::Transpose);

        auto a_trans = a_trans_op->compute({inputs[0]});
        auto b_trans = b_trans_op->compute({inputs[1]});
        a_trans->compact();
        b_trans->compact();

        std::shared_ptr<GenericOp<Dtype>> a_gemm_op = 
            std::make_shared<MatMulOp<Dtype>>(OpType::MatMul);
        std::shared_ptr<GenericOp<Dtype>> b_gemm_op = 
            std::make_shared<MatMulOp<Dtype>>(OpType::MatMul);

        auto a_grad = a_gemm_op->compute({out_grad, b_trans});
        auto b_grad = b_gemm_op->compute({a_trans, out_grad});

        if(a_org_size < a_grad->size()) {
            a_grad = __grad_sum(a_grad, a_org_shape);
        }

        if(b_org_size < b_grad->size()) {
            b_grad = __grad_sum(b_grad, b_org_shape);
        }

        return {a_grad, b_grad};
    }

protected:
    virtual inline cudaError_t _get_num_blocks() override {
        int dev, sm_count, tpm;
        cudaError err = __get_gpu_info(&dev, &sm_count, &tpm);
        _num_blocks = std::max<int>(1, std::min<int64_t>((_n + kBlockSize - 1) / kBlockSize,
                                               sm_count * tpm / kBlockSize * NUMWAVES));
        return cudaSuccess;

    }

private:
    cached_data_type __grad_sum(cached_data_type grad, std::vector<int32_t> org_shape) {
        auto grad_shape = grad->shape();
        int shape_diff = grad_shape.size() - org_shape.size();
        std::vector<int> sum_axes;

        if (shape_diff>0) {
            for(int i=0; i<shape_diff; ++i)
                sum_axes.push_back(i);
        } else {
            for(int i=0; i<grad_shape.size(); ++i) {
                if(grad_shape[i] > org_shape[i])
                    sum_axes.push_back(i);
            }
        }

        std::shared_ptr<GenericOp<Dtype>> sum_op = 
            std::make_shared<SummationOp<Dtype>>(sum_axes, OpType::Summation);

        return sum_op->compute({grad});
    }

    std::vector<int> __calc_trans_axes(std::vector<int32_t> shape) {
        int shape_lenth = shape.size();
        std::vector<int> axes={shape_lenth-2, shape_lenth-1};

        return axes;
    }

    int __broadcast_axes(std::vector<int32_t> input_shape1, 
                         std::vector<int32_t> input_shape2) {
        int idx = -1;
        int input1_length = input_shape1.size();
        int input2_length = input_shape2.size();

        if(input1_length > input2_length) {
            for(int i=0; i<input1_length-2; ++i)
                _bshape.push_back(input_shape1[i]);
            _bshape.push_back(input_shape2[input2_length-2]);
            _bshape.push_back(input_shape2[input2_length-1]);

            __calc_broadcast_axes(input_shape1, input_shape2);
            idx = 1;
        } else if(input1_length < input2_length) {

            for(int i=0; i<input2_length-2; ++i)
                _bshape.push_back(input_shape2[i]);
            _bshape.push_back(input_shape1[input1_length-2]);
            _bshape.push_back(input_shape1[input1_length-1]);

            __calc_broadcast_axes(input_shape2, input_shape1);
            idx = 0;
        } else {
            int sum_shape1 = std::accumulate(input_shape1.begin(), input_shape1.end()-2, 0);
            int sum_shape2 = std::accumulate(input_shape2.begin(), input_shape2.end()-2, 0);

            if(sum_shape1 > sum_shape2) {
                for(int i=0; i<input1_length-2; ++i)
                    _bshape.push_back(input_shape1[i]);
                _bshape.push_back(input_shape2[input2_length-2]);
                _bshape.push_back(input_shape2[input2_length-1]);

                __calc_broadcast_axes(input_shape1, input_shape2);
                idx = 1;
            } else if (sum_shape1 < sum_shape2){
                for(int i=0; i<input2_length-2; ++i)
                    _bshape.push_back(input_shape2[i]);
                _bshape.push_back(input_shape1[input1_length-2]);
                _bshape.push_back(input_shape1[input1_length-1]);

                __calc_broadcast_axes(input_shape2, input_shape1);
                idx = 0;
            }
        }

        return idx;
    }

    void __calc_broadcast_axes(std::vector<int32_t>& bigger_shape, 
                               std::vector<int32_t>& small_shape) {
        int size_diff = bigger_shape.size() - small_shape.size();
        for(int i=0; i<size_diff; ++i)
            _baxes.push_back(i);

        for(int i=small_shape.size()-1; i>=0; --i) {
            if(small_shape[i]==1 && _bshape[i+size_diff]>1) {
                _baxes.push_back(i+size_diff);
            } else if(small_shape[i]>1 && _bshape[i+size_diff]!= small_shape[i]) {
                assert(true && "broadcast shape does not match");
            } 
        }
    }

    std::vector<int32_t> __prepare_shape(std::vector<int32_t> input_shape) {
        int shape_length = input_shape.size();
        //if(shape_length==2) return input_shape;

        std::vector<int32_t> out_shape(3);

        int first_shape = 1;
        for(int i=0; i<shape_length-2; ++i)
            first_shape *= input_shape[i];

        out_shape[0] = first_shape;
        out_shape[1] = input_shape[shape_length-2];
        out_shape[2] = input_shape[shape_length-1];

        return out_shape;
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

    std::vector<int> _baxes;
    std::vector<int32_t> _bshape;
};

#endif

