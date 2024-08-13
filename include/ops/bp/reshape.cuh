#ifndef __RESHAPE_OP__
#define __RESHAPE_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class ReshapeOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    ReshapeOp(std::vector<int32_t> new_shape, OpType op_type):
        GenericOp<Dtype>(op_type), _new_shape(new_shape) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size() == 1 && "number of reshape input must be 1");

        /* awkward... is there any better idea not compacting in here */
        inputs[0]->compact();

        cached_data_type cached_data = __create_cached_data(_new_shape, inputs[0]->dtype,
                                                            inputs[0]->device(), false);
        /* without deep cpy data, reuse cached data in inputs[0] */
        cached_data->array = inputs[0]->array;
        cached_data->set_offset(inputs[0]->offset());

        cached_data->cached = true;
        cached_data->is_compact = true;
        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor) override {

        cached_data_type out = out_grad->deep_cpy_cached_data();
        out->set_shape(tensor->inputs[0]->shape());
        out->compact_strides();

        out->cached = true;
        out->is_compact = true;

        return {out};
    }

private:
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
    std::vector<int32_t> _new_shape;
};

#endif

