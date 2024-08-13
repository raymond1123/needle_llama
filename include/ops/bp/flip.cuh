#ifndef __FLIP_OP__
#define __FLIP_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class FlipOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    FlipOp(OpType op_type, std::vector<int> axes):
        GenericOp<Dtype>(op_type), _axes(axes), _offset(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size() == 1 && "number of reshape input must be 1");

        cached_data_type cached_data = __create_cached_data(inputs[0]->shape(), inputs[0]->dtype,
                                                            inputs[0]->device(), false);

        /* without deep cpy data, reuse cached data in inputs[0] */
        cached_data->array = inputs[0]->array;

        __calc_param(inputs[0]->shape(), 
                     inputs[0]->strides(),
                     inputs[0]->offset());

        cached_data->set_offset(_offset);
        cached_data->set_strides(_new_strides);

        cached_data->cached = true;
        cached_data->is_compact = false;

        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor) override {

        std::shared_ptr<GenericOp<Dtype>> flip_op = 
            std::make_shared<FlipOp<Dtype>>(OpType::Flip, _axes);

        cached_data_type out = flip_op->compute({out_grad});
        out->compact(); // compact here before reduced add

        return {out};
    }

private:
    void __calc_param(std::vector<int32_t> shape, 
                      std::vector<int32_t> strides,
                      size_t offset) {

        int length_shape = shape.size();
        _new_strides = strides;

        for(auto& axis: _axes) {
            if(axis<0) axis += length_shape;
            _new_strides[axis] *= -1;
            _offset += (shape[axis]-1)*strides[axis];
        }

        _offset += offset;
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
    std::vector<int32_t> _new_strides;
    size_t _offset;

};

#endif

