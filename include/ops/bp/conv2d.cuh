#ifndef __CONV2D_OP__
#define __CONV2D_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class iTanh {
public:
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               Dtype* out) {
        if constexpr (std::is_same<Dtype, __half>::value)
            __half_tanh(n, a, out);
        else 
            __fp32_tanh(n, a, out);
    }

private:
    __device__ void __half_tanh(size_t n,
                               const Dtype* a, 
                               Dtype* out) {
        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx < n)
            out[tx] = __float2half(tanhf(__half2float(a[tx])));
    }

    __device__ void __fp32_tanh(size_t n,
                               const Dtype* a, 
                               Dtype* out) {
        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if (tx < n) 
            out[tx] = tanhf(a[tx]);
    }
};

template<typename Dtype>
class iTanhGrad {
public:
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               const Dtype* b, 
                               Dtype* out) {

        size_t tx = blockDim.x*blockIdx.x+threadIdx.x;
        if(tx >=n) return;
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
iApplyTanh(size_t n, const Dtype* a, Dtype* out) {
    auto functor = iTanh<Dtype>();
    functor(n, a, out);
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
iApplyTanhGrad(size_t n, const Dtype* a, const Dtype* b, Dtype* out) {
    auto functor = iTanhGrad<Dtype>();
    functor(n, a, b, out);
}

template<typename Dtype>
class Conv2dOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    /* TODO: group & dilate */
    Conv2dOp(OpType op_type, int stride, int padding): 
        GenericOp<Dtype>(op_type), _stride(stride), _padding(padding) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        /*
            img: inputs[0], kernel: inputs[1]

            different from pytorch
            input_shape:  [batch_size,  height,      width,       in_channel]
            kernel_shape: [kernel_size, kernel_size, in_channel,  out_channel]
            out_shape:    [batch_size,  height,      width,       out_channel]
        */
        _img_shape = inputs[0]->shape(); // [2,4,3,3]
        _img_stride = inputs[0]->strides();
        _kernel_shape = inputs[1]->shape();

        assert(inputs.size()==2 && "input number of Conv2dOp must be 2");
        assert(_img_shape.size()==4 && "size of conv kernel shape must be 4");
        assert(_kernel_shape.size()==4 && "size of input shape must be 4");

        inputs[0]->compact();

        /* padding first */
        std::vector<int32_t> padding_axes = {0,0, _padding,_padding, _padding,_padding, 0,0};
        std::shared_ptr<GenericOp<Dtype>> padding_op = 
            std::make_shared<PaddingOp<Dtype>>(OpType::Padding, padding_axes, 0);
        cached_data_type pad_cache = padding_op->compute({inputs[0]});

        /* TODO */
        assert(_kernel_shape[0]==_kernel_shape[1] && "kernel size must be equal");

        /* need to use padding shape and strides */
        _img_shape = pad_cache->shape(); // [2,6,5,3]
        _img_stride = pad_cache->strides();
        __calc_shape();

        /* im2col as_stride */
        cached_data_type im2col = pad_cache;
        im2col->set_shape(_im2col_shape); // [2,4,3,3,3,3]
        im2col->set_strides(_im2col_stride);
        im2col->is_compact = false;
        im2col->cached = true;

        /* transform conv2d to matmul */
        im2col->compact();
        im2col->set_shape(_im2col_matmul_shape);

        cached_data_type kernel = inputs[1];
        kernel->compact();
        kernel->set_shape(_kernel_matmul_shape);

        std::shared_ptr<GenericOp<Dtype>> matmul_op= 
            std::make_shared<MatMulOp<Dtype>>(OpType::MatMul);
        cached_data_type out = matmul_op->compute({im2col, kernel});
        out->compact();
        out->set_shape(_out_shape);

        out->cached = true;
        out->is_compact = true;

        return out;
    }

    /* TODO */
    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {
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

    inline void __calc_shape() {
        int batch_size = _img_shape[0];
        int height = _img_shape[1];
        int width = _img_shape[2];
        int channel = _img_shape[3];

        int kernel_size = _kernel_shape[0];
        int c_in = _kernel_shape[2];
        int c_out = _kernel_shape[3];

        int out_height = (height-kernel_size+1)/_stride;
        int out_width = (width-kernel_size+1)/_stride;

        _out_shape = {batch_size, out_height, out_width, c_out};

        _im2col_shape = {batch_size, out_height, out_width, 
                         kernel_size, kernel_size, c_in};

        _im2col_stride = {_img_stride[0], 
                          _img_stride[1]*_stride, 
                          _img_stride[2]*_stride, 
                          _img_stride[1],
                          _img_stride[2],
                          _img_stride[3]};

        _im2col_matmul_shape = {batch_size*out_height*out_width, 
                                kernel_size*kernel_size*c_in};
        _kernel_matmul_shape = {kernel_size*kernel_size*c_in, c_out};
    }

private:
    size_t _n;
    std::vector<int32_t> _kernel_shape;
    std::vector<int32_t> _kernel_matmul_shape;
    std::vector<int32_t> _img_shape;
    std::vector<int32_t> _out_shape;

    std::vector<int32_t> _img_stride;
    std::vector<int32_t> _im2col_shape;
    std::vector<int32_t> _im2col_stride;
    std::vector<int32_t> _im2col_matmul_shape;

    int _stride;
    int _padding;

    int _num_blocks;
    cached_data_type _idx_ptr;
};

#endif

