#pragma once

#include "common.hpp"
#include "nn/nn_module.cuh"
#include "init/initial.hpp"
#include <cublas_v2.h>

namespace py = pybind11;

class ModuleList {
public:
    ModuleList() = default;

    // 添加模块的方法
    void append(std::shared_ptr<Module> module) {
        modules.push_back(module);
    }

    // 迭代器支持，使其行为类似于一个容器
    auto begin() { return modules.begin(); }
    auto end() { return modules.end(); }

    // 访问指定索引的模块
    std::shared_ptr<Module> operator[](size_t index) {
        return modules.at(index);
    }

    // 获取模块的数量
    size_t size() const {
        return modules.size();
    }

private:
    std::vector<std::shared_ptr<Module>> modules;
};
