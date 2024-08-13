#ifndef __OPTIMIZE_HPP__
#define __OPTIMIZE_HPP__

#include "nn/nn_module.cuh"

template<typename Dtype>
class Optimizer {
public:
    using param_type = std::shared_ptr<Tensor<Dtype>>;

    Optimizer(std::vector<param_type>& params, float lr=0.01): 
        _params(params), _lr(lr) {}

    void reset_grad() {
        for(auto& w: this->_params)
            w->grad.reset();
    }

    virtual void step() {}

protected:
    std::vector<param_type> _params;
    float _lr;
};

template<typename Dtype>
class SGD: public Optimizer<Dtype> {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

    SGD(std::vector<param_type>& params, float lr=0.01, 
        float weight_decay=0.0, float momentum=0.0): 
        Optimizer(params, lr), _weight_decay(weight_decay) {}

    virtual void step() override {

        for(auto& w: this->_params) {
            cached_data_type grad = _weight_decay>0.0?w->grad + _weight_decay*(*w):w->grad;
            Tensor<Dtype> tmp = ((*w)*_momentum + (*grad)*(1-_momentum))*_lr;
            (*w) -= tmp*_lr;
        }
    }

private:
    float _momentum;
    float _weight_decay;
    //std::unordered_map<param_type, std::vector<param_type>> _update;
};

// TODO
template<typename Dtype>
class Adam: public Optimizer<Dtype> {

};

#endif

