#include "TensorTransform.h"
#include "Tensor.h"
#include <vector>
#include <cstdlib>      // rand
#include <ctime>        // time
#include <stdexcept>    // para usar invalid_argument

using namespace std;


/*
Metodos de clase para crear tensores especiales
    zeros: crea un tensor con todas sus posiciones inicializadas en cero.
    ones: crea un tensor con todas sus posiciones inicializadas en uno.
    random: crea un tensor con valores aleatorios distribuidos uniformemente en el rango
           [min,max).
    arange: crea un tensor unidimensional con valores secuenciales desde start hasta end
           (no inclusivo).
*/


Tensor Tensor::zeros(const vector<size_t>& shape) {
    size_t total = 1;
    for (auto d : shape) {
        total *= d;
    }

    vector<double> values(total, 0.0);
    return Tensor(shape, values);
}

Tensor Tensor::ones(const vector<size_t>& shape) {

    size_t total = 1;
    for (auto d : shape) {
        total *= d;
    }

    vector<double> values(total, 1.0);

    return Tensor(shape, values);
}

Tensor Tensor::random(const vector<size_t>& shape, double min, double max) {
    size_t total = 1;
    for (auto d : shape) {
        total *= d;
    }

    vector<double> values(total);

    for (size_t i = 0; i < total; i++) {
        double r = (double) rand() / RAND_MAX; // 0 a 1
        values[i] = min + r * (max - min);     // escalar
    }

    return Tensor(shape, values);
}

Tensor Tensor::arange(int start, int end) {

    vector<double> values;

    for (int i = start; i < end; i++) {
        values.push_back(i);
    }

    return Tensor({values.size()}, values);
}

Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}


//Sobrecarga de operadores
Tensor Tensor::operator+(const Tensor& other) const {

    if (dimensiones != other.dimensiones)
        throw std::invalid_argument("Dimensiones incompatibles");

    vector<double> result(size);

    for (size_t i = 0; i < size; i++) {
        result[i] = data[i] + other.data[i];
    }

    return Tensor(dimensiones, result);
}

Tensor Tensor::operator-(const Tensor& other) const {

    if (dimensiones != other.dimensiones)
        throw std::invalid_argument("Dimensiones incompatibles");

    vector<double> result(size);

    for (size_t i = 0; i < size; i++) {
        result[i] = data[i] - other.data[i];
    }

    return Tensor(dimensiones, result);
}

Tensor Tensor::operator*(const Tensor& other) const {

    if (dimensiones != other.dimensiones)
        throw std::invalid_argument("Dimensiones incompatibles");

    vector<double> result(size);

    for (size_t i = 0; i < size; i++) {
        result[i] = data[i] * other.data[i];
    }

    return Tensor(dimensiones, result);
}

Tensor Tensor::operator*(double scalar) const {

    vector<double> result(size);

    for (size_t i = 0; i < size; i++) {
        result[i] = data[i] * scalar;
    }

    return Tensor(dimensiones, result);
}