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
// en Tensor.cpp - implementacion
Tensor::Tensor(const vector<size_t>& shape, const vector<double>& values) {
    if (shape.size() > 3) {
        throw invalid_argument("Maximo 3 dimensiones");
    }

    dimensiones = shape;

    size = 1;
    for (auto d : shape) {
        size *= d;
    }

    if (values.size() != size) {
        throw invalid_argument("El numero de valores no coincide con las dimensiones");
    }

    data = new double[size];
    for (size_t i = 0; i < size; i++) {
        data[i] = values[i];
    }
}

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

Tensor Tensor::view(const vector<size_t>& nueva_forma) {
    // calculamos cuantos elementos tendria nuestro nuevo tensor
    size_t total_nuevo = 1;
    for (size_t i = 0; i < nueva_forma.size(); i++) {
        total_nuevo *= nueva_forma[i];
    }

    // tiene que coicidir con el tamano actual, si no lo hace hay un error
    if (total_nuevo != size) {
        throw invalid_argument("Error,  los elementos no coinciden con la nueva forma");
    }

    // no puede haber mas de 3 dimensiones
    if (nueva_forma.size() > 3) {
        throw invalid_argument("Error, no estan permitidas + de 3 dimensiones");
    }

    //  construimos el nuevo tensor (se copian los datos en un vector)
    vector<double> copia_datos(data, data + size);
    return Tensor(nueva_forma, copia_datos);
    // el return mueve  el resultado , no hay copia extra
}

Tensor Tensor::unsqueeze(size_t axis) {
    if (dimensiones.size() >= 3) {
    throw invalid_argument("invalido, a existen 3 dimensiones");
    }
    if (axis > dimensiones.size()) {
        throw invalid_argument("El eje esta fuera del rango válido");
        }

    // incertamos una dimension de tamaño 1 en la posicion indicada
    vector<size_t> forma_nueva(dimensiones);
    forma_nueva.insert(forma_nueva.begin() + axis, (size_t)1);

    // recalculamnos la forma lógica del tensor
    vector<double> mismos_datos(data, data + size);
    return Tensor(forma_nueva, mismos_datos);
}

//concat va a unir tensores a lo largo de un eje
Tensor Tensor::concat(const vector<Tensor>& tensores, size_t axis) {
    if (tensores.size() == 0) {
        throw invalid_argument("no hay tensores para unir");
    }

    size_t num_dims = tensores[0].dimensiones.size();

    // revisamos que todos tengan la misma cantidad de dimensiones
    for (size_t i = 1; i < tensores.size(); i++) {
        if (tensores[i].dimensiones.size() != num_dims) {
            throw invalid_argument("¡advertencia: dimensiones distintas entre tensores!");
        }
        // y que en cada dimension coincidan a excepcion del axis o eje
        for (size_t d = 0; d < num_dims; d++) {
            if (d == axis) continue;
            if (tensores[i].dimensiones[d] != tensores[0].dimensiones[d]) {
                throw invalid_argument("los tamanios no son compatibles");
            }
        }
    }

    if (axis >= num_dims) {
        throw invalid_argument("eje fuera de rango");
    }

    // construimos la nueva forma sumando el eje de concatenación
    vector<size_t> forma_resultado = tensores[0].dimensiones;
    for (size_t i = 1; i < tensores.size(); i++) {
        forma_resultado[axis] += tensores[i].dimensiones[axis];
    }

    // acumulamos todos los datos
    vector<double> datos_totales;
    for (size_t i = 0; i < tensores.size(); i++) {
        for (size_t j = 0; j < tensores[i].size; j++) {
            datos_totales.push_back(tensores[i].data[j]);
            }
        }

    // construimos y retornamos con move
    Tensor res(forma_resultado, datos_totales);
    return res;
}


Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.size != b.size) {
        throw invalid_argument("DOT:  los tensores no tienen el mismo numero de elementos");
    }

    double acumulado = 0.0;
    for (size_t i = 0; i < a.size; i++) {
        acumulado += a.data[i] * b.data[i];
        }

    vector<double> res_vec = {acumulado};
    return Tensor({1}, res_vec);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dimensiones.size() != 2 || b.dimensiones.size() != 2) {
        throw invalid_argument("MALMUT: se necesitan dos matrices 2D");
        }

    size_t filas_a = a.dimensiones[0];
    size_t cols_a  = a.dimensiones[1];
    size_t filas_b = b.dimensiones[0];
    size_t cols_b  = b.dimensiones[1];

    // para poder multiplicar, columnas de A deben ser iguales a filas de B
    if (cols_a != filas_b) {
        throw invalid_argument("matmul: columnas de A deben coincidir con filas de B");
    }

    
    size_t total = filas_a * cols_b;

    vector<double> datos_resultado(total, 0.0);

    for (size_t i = 0; i < filas_a; i++) {
        for (size_t j = 0; j < cols_b; j++) {
            double suma_parcial = 0.0;

            for (size_t p = 0; p < cols_a; p++) 
            {
                suma_parcial += a.data[i * cols_a + p] * b.data[p * cols_b + j];
        }
            datos_resultado[i * cols_b + j] = suma_parcial;
        }
    }

    return Tensor({filas_a, cols_b}, datos_resultado);
    }
