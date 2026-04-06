#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class TensorTransform; // forward declaration
class Tensor {
private:
    vector<size_t> dimensiones;
    double* data = nullptr;
    size_t size;

public:
    Tensor(const vector<size_t>& shape, const vector<double>& values);

    //Copiar constructor
    Tensor(const Tensor& other) {
        dimensiones = other.dimensiones;
        size = other.size;

        data = new double[size];

        for (size_t i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
    }

    //Copiar assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {

            delete[] data;

            dimensiones = other.dimensiones;
            size = other.size;

            data = new double[size];

            for (size_t i = 0; i < size; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    //Mover el constructor
    Tensor(Tensor&& other) noexcept {
        dimensiones = other.dimensiones;
        size = other.size;
        data = other.data;

        other.data = nullptr;
    }

    //Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {

            delete[] data;

            dimensiones = other.dimensiones;
            size = other.size;
            data = other.data;

            other.data = nullptr;
        }
        return *this;
    }

    //Destructor
    ~Tensor() {
        delete[] data;
    }


    //Metodo apply
    Tensor apply(const TensorTransform& transform) const;

    //Getters
    const vector<size_t>& getShape() const { 
        return dimensiones; 
    }
    double* getData() const { 
        return data; 
    }
    size_t getSize() const { 
        return size; 
    }

    //Aplicamos metodos de clase para crear tensores especiales
    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape, double min, double max);
    static Tensor arange(int start, int end);

    //Sobrecarga de operadores
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;


};


#endif