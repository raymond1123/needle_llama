#ifndef __PERMUTE_OP__
#define __PERMUTE_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class PermuteOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    PermuteOp(std::vector<int> axes, OpType op_type):
        GenericOp<Dtype>(op_type), _axes(axes) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size() == 1 && "number of reshape input must be 1");
        assert(inputs[0]->shape().size() == _axes.size() &&
               "number of reshape input must be 1");

        __prepare_pos_axes(inputs[0]->shape(), inputs[0]->strides());
        cached_data_type cached_data = __create_cached_data(_new_shape,
                                                            inputs[0]->dtype,
                                                            inputs[0]->device(), false);
        /* without deep cpy data, reuse cached data in inputs[0] */
        cached_data->array = inputs[0]->array;
        cached_data->set_strides(_new_strides);
        cached_data->set_offset(inputs[0]->offset());

        cached_data->cached = true;
        cached_data->is_compact = false;
        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor) override {

        auto inputs = tensor->inputs;
        int shape_length = inputs[0]->shape().size();
        std::vector<int> paxes(shape_length);

        for(int i=0; i<shape_length; ++i)
            paxes[i] = _axes[_axes[i]];

        std::shared_ptr<GenericOp<Dtype>> permute_op = 
            std::make_shared<PermuteOp<Dtype>>(paxes, OpType::Permute);
        cached_data_type out = permute_op->compute({out_grad});

        out->compact(); // compact here before reduced add
        return {out};
    }

private:
    void __prepare_pos_axes(std::vector<int32_t> shape,
                            std::vector<int32_t> strides) {
        int length_shape = shape.size();
        _new_strides = strides;
        _new_shape = shape;

        auto pos_axes = _axes;

        for(int i=0; i<_axes.size(); ++i)
            if(_axes[i]<0) pos_axes[i] = length_shape+_axes[i];

        for(int i=0; i<length_shape; ++i) {
            _new_strides[i] = strides[pos_axes[i]];
            _new_shape[i] = shape[pos_axes[i]];
        }
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

protected:
    virtual inline cudaError_t _get_num_blocks() override {
        return cudaSuccess;
    }

private:
    std::vector<int> _axes;
    std::vector<int32_t> _new_shape;
    std::vector<int32_t> _new_strides;
};

#endif

