#include "TensorTransform.h"
#include "Tensor.h"
#include <iostream>



int main(){
    Tensor A = Tensor::zeros({2,3});
    Tensor B = Tensor::ones({3,3});
    Tensor C = Tensor::random({2,2}, 0.0, 1.0);
    Tensor D = Tensor::arange(0,6);

    Tensor E = A + B;
    Tensor F = A - B;
    Tensor G = A * B;
    Tensor H = A * 2.0;

    ReLU relu;
    Sigmoid sig;

}
