#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <filesystem>

#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdarg.h>

#include <stdexcept>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include <iterator>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

enum class BackendType: int {
    CPU = 0,
    CUDA = 1,
    NOT_CONCERN=2,
};

enum class MemCpyType: int {
    Host2Host = 0,
    Hosta2Hostb = 1,
    Host2Dev = 2,
    Dev2Host = 3,
    Dev2Dev = 4
};

enum class OpType: int {
    LAEF=-1,
    EWAddTensor = 0,
    EWAddScalar = 1,
    EWMinusTensor = 2,
    EWMinusScalar = 3,
    EWMulTensor = 4,
    EWMulScalar = 5,
    EWDivTensor = 6,
    EWDivScalar = 7,
    EWPowTensor = 8,
    EWPowScalar = 9,
    MatMul = 10,
    Padding = 11,
    Neg = 12,
    Log = 13,
    Exp = 14,
    Relu = 15,
    Tanh = 16,
    Reshape = 17,
    BroadcastTo = 18,
    Transpose = 19,
    Permute = 20,
    Summation = 21,
    Max = 22,
    Slice = 23,
    Setitem = 24,
    Flip = 25,
    Dilate = 26,
    Undilate = 27,
    Stack = 27,
    Split = 29,
    LogSumExp= 30,
    Conv = 31,
    RoTri = 32
};

enum class DataType: int{
    FLOAT=0, HALF=1, 
    NOT_CONCERN=2,
};

enum class ModuleOp: int {
    TRAIN = 0,
    VAL = 1,
    TO_HALF = 2,
};

#endif

