#ifndef __TRANSPOSE_OP__
#define __TRANSPOSE_OP__

#include "ops/generic_op.cuh"
#include "ops/bp/permute.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class TransposeOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    TransposeOp(std::vector<int> axes, OpType op_type):
        GenericOp<Dtype>(op_type), _axes(axes) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size() == 1 && "number of reshape input must be 1");
        assert(_axes.size() == 2 && "number of transpose axes must be 2");

        auto permute_axes = __prepare_pos_axes(inputs[0]->shape().size());

        std::shared_ptr<GenericOp<Dtype>> permute_op = 
            std::make_shared<PermuteOp<Dtype>>(permute_axes, OpType::Permute);

        cached_data_type cached_data = permute_op->compute({inputs[0]});

        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor) override {

        cached_data_type out = compute({out_grad});
        return {out};
    }

private:
    inline std::vector<int> __prepare_pos_axes(int shape_length) {
        auto pos_axes = _axes;
        std::vector<int> permute_axes(shape_length);

        for(int i=0; i<shape_length; ++i)
            permute_axes[i] = i;

        int pos_axis_1 = _axes[0], pos_axis_2 = _axes[1];
        if(_axes[0]<0) pos_axis_1 = shape_length + _axes[0];
        if(_axes[1]<0) pos_axis_2 = shape_length + _axes[1];

        int tmp = permute_axes[pos_axis_1];
        permute_axes[pos_axis_1] = pos_axis_2;
        permute_axes[pos_axis_2] = tmp;

        return permute_axes;
    }

protected:
    virtual inline cudaError_t _get_num_blocks() override {
        return cudaSuccess;
    }

private:
    std::vector<int> _axes;
};

#endif

