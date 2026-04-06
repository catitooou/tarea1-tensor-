#ifndef TENSOR_TRANSFORM_H
#define TENSOR_TRANSFORM_H

#include "Tensor.h"
#include <cmath>
#include <vector>

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

// 🔹 ReLU
class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {

        std::vector<double> result(t.getSize());

        for (size_t i = 0; i < t.getSize(); i++) {
            result[i] = std::max(0.0, t.getData()[i]);
        }

        return Tensor(t.getShape(), result);
    }
};

// 🔹 Sigmoid
class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {

        std::vector<double> result(t.getSize());

        for (size_t i = 0; i < t.getSize(); i++) {
            double x = t.getData()[i];
            result[i] = 1.0 / (1.0 + std::exp(-x));
        }

        return Tensor(t.getShape(), result);
    }
};




#endif