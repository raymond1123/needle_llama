#ifndef __SLICE_OP__
#define __SLICE_OP__

#include "ops/generic_op.cuh"
#include "backend/cuda_util.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class SliceOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    SliceOp(OpType op_type, std::vector<py::object>& indices, 
            std::vector<int32_t> shape, std::vector<int32_t> strides,
            size_t offset):
        GenericOp<Dtype>(op_type), _indices(indices), 
        _org_shape(shape), _org_strides(strides),
        _new_shape(std::vector<int32_t>(shape.size())), 
        _new_strides(std::vector<int32_t>(shape.size())), 
        _offset(offset), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of SliceOp must be 1");

        __slice();
        cached_data_type cached_data = __create_cached_data(_new_shape, inputs[0]->dtype,
                                                            inputs[0]->device(), false);

        /* without deep cpy data, reuse cached data in inputs[0] */
        cached_data->array = inputs[0]->array;
        cached_data->set_strides(_new_strides);
        cached_data->set_offset(_offset);

        cached_data->cached = true;
        cached_data->is_compact = false;

        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {
        cached_data_type out_cached = __create_cached_data(_org_shape, out_grad->dtype,
                                                           out_grad->device());
        //_n = out_cached->size();
        out_cached->zeros();

        std::shared_ptr<GenericOp<Dtype>> slice_op = 
            std::make_shared<SliceOp<Dtype>>(OpType::Slice, _indices, 
                                            out_cached->shape(), 
                                            out_cached->strides(), 
                                            out_cached->offset());

        auto view = slice_op->compute({out_cached});

        _n = view->size();
        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in SliceOp failed");

        ApplyEwSetitem<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                  out_grad->cached_ptr(), 
                                                  view->cached_ptr(), 
                                                  VecToCuda(view->shape()), 
                                                  VecToCuda(view->strides()), 
                                                  view->offset());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEwSetitem failed");
        out_cached->compact(); // compact here before reduced add

        return {out_cached};
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
    void __slice() {
        size_t num_dims = _org_shape.size();
        int ellipsis_cnt = 0; 
        int ellipsis_start = -1, ellipsis_end = -1, ellipsis_index = -1;

        std::vector<py::slice> slices;

        // traverse indices, find number of '...' in slice
        for(size_t i = 0; i < _indices.size(); ++i) {
            if (py::isinstance<py::ellipsis>(_indices[i])) {
                ellipsis_index = i;
                ellipsis_start = i;
                ellipsis_cnt += 1;
            }
        }

        assert(ellipsis_cnt<=1 && "number of ellipsis in slice can only be 1");
        ellipsis_end = num_dims-(_indices.size()-ellipsis_index-1)-1;

        int slice_index = 0;
        bool ellipsis_add1 = false;

        for(int i=0; i<num_dims; ++i) {
            if(ellipsis_cnt>0 and i>=ellipsis_start and i<=ellipsis_end) {
                slices.push_back(py::slice(0, _org_shape[i], 1));
                if(!ellipsis_add1) { 
                    slice_index++;
                    ellipsis_add1=true; 
                }
            } else {
                if(slice_index >= _indices.size()) {
                    slices.push_back(py::slice(0, _org_shape[i], 1));
                } else {
                    auto slice = py::cast<py::slice>(_indices[slice_index]);
                    slices.push_back(slice);
                }

                slice_index++;
            }
        }

        for(int i=0; i<slices.size(); ++i) {
            auto pos_slice = __process_slice(slices[i], _org_shape[i]);
            _pos_slices.push_back(pos_slice);
            _new_shape[i] = static_cast<size_t>(pos_slice[3]);
            assert(_new_shape[i]>0 && "slice failed");
        }

        for(int i=0; i<num_dims; ++i) {
            _new_strides[i] = _org_strides[i]*_pos_slices[i][2];
            _offset += _org_strides[i]*_pos_slices[i][0];
        }
    }

    std::vector<Py_ssize_t> __process_slice(py::slice slice, size_t length) {
        Py_ssize_t start, stop, step, slicelength;
        if (!slice.compute(length, &start, &stop, &step, &slicelength)) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to extract slice indices");
            throw std::runtime_error("Failed to extract slice indices");
            return {};
        }

        return {start, stop, step, slicelength};
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

    std::vector<py::object> _indices;
    std::vector<int32_t> _org_shape;
    std::vector<int32_t> _org_strides;

    std::vector<int32_t> _new_shape;
    std::vector<int32_t> _new_strides;
    size_t _offset;

    std::vector<std::vector<Py_ssize_t>> _pos_slices;
};

#endif

