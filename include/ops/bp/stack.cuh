#ifndef __STACK_OP__
#define __STACK_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/setitem.cuh"
#include "ops/bp/split.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class StackOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    StackOp(int dim, OpType op_type):
        GenericOp<Dtype>(op_type), _dim(dim), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()>1 && "input number of SummationOp must bigger than 1");
        for(auto& input: inputs) {
            _shapes.push_back(input->shape());
            input->compact();
        }

        int shape_length = _shapes[0].size();
        int concated_dim = shape_length+1;
        __calc_new_shape(inputs.size(), shape_length, concated_dim);
        cached_data_type cached_out = __create_cached_data(_new_shape,
                                                           inputs[0]->dtype,
                                                           inputs[0]->device());
        for(int i=0; i<inputs.size(); ++i)
            __stack_single_tensor(cached_out, inputs[i], i, concated_dim);

        cached_out->cached = true;
        cached_out->is_compact= true;

        return cached_out;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {

        auto inputs = tensor->inputs;
        int split_size = out_grad->shape()[_dim];
        std::vector<cached_data_type> out_cached;

        for(int i=0; i<split_size; ++i) {
            std::shared_ptr<GenericOp<Dtype>> split_op =
                std::make_shared<SplitOp<Dtype>>(_dim, i, OpType::Split);

            out_cached.push_back(split_op->compute({out_grad}));
            out_cached[i]->compact();
        }

        return out_cached;
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
    void __stack_single_tensor(cached_data_type cached_out,
                               cached_data_type input, 
                               int i, int concated_dim) {

        std::vector<py::object> indices;

        for(int j=0; j<concated_dim; ++j) {
            if(j==_dim) indices.push_back(py::slice(i,i+1,1));
            else indices.push_back(py::slice(0, _new_shape[j], 1));
        }

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, indices, 
                                             cached_out->shape(), 
                                             cached_out->strides(), 
                                             cached_out->offset());

        auto view = slice_op->compute({cached_out});

        _n = view->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        ApplyEwSetitem<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  input->cached_ptr(), 
                                                  view->cached_ptr(), // out
                                                  VecToCuda(view->shape()), 
                                                  VecToCuda(view->strides()), 
                                                  view->offset());

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEwSetitem failed");
    }

    void __calc_new_shape(int input_size, 
                          int shape_length, 
                          int concated_dim) {

        if(_dim<0) _dim += concated_dim;
        assert((0<=_dim && _dim<concated_dim) && "dim not in the right scope");

        for(int i=1; i<_shapes.size(); ++i) {
            assert(_shapes[i].size()==_shapes[i-1].size() && "stack shapes not the same");

            for(int j=0; j<_shapes[i].size(); ++j)
                assert(_shapes[i][j] == _shapes[i-1][j] && "stack shapes not the same");
        }

        _new_shape = _shapes[0];
        _new_shape.insert(_new_shape.begin()+_dim, input_size);

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

    int _dim;
    std::vector<int32_t> _new_shape;
    std::vector<std::vector<int32_t>> _shapes;
};

#endif

