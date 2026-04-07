# tarea1-tensor-
Buenos días, este es el repositorio de la tarea 01 del curso de programación 3 con el profesor de laboratorio José Chávez 

#**Proyecto Tensores**
Este proyecto implementa una clase Tensor en C++ que permite manejar estructuras multidimensionales, incluyendo operaciones matemáticas y transformaciones como ReLU y Sigmoid.

Compilación
Para compilar el código es necesario usar este comando:
Compilar y ejecutar :)
g++ main.cpp Tensor.cpp -o main
./main
## Archivos
- `Tensor.h` → declaraciones de la clase Tensor
- `Tensor.cpp` → implementaciones
- `TensorTransform.h` → interfaz abstracta + ReLU + Sigmoid
- `main.cpp` → pruebas y red neuronal

## Lo que implementa cada integrante

### Integrante Keyra Huamanyauri Alvarado
- Constructor principal y gestión de memoria
- Tensores predefinidos: zeros, ones, random, arange
- Sobrecarga de operadores: +, -, *, * escalar
- Polimorfismo: ReLU y Sigmoid

### Integrante Catherine Lopez Chavez
- view: reorganiza dimensiones sin copiar datos
- unsqueeze: agrega dimensión de tamaño 1
- concat: une tensores a lo largo de un eje
- dot: producto punto entre tensores
- matmul: multiplicación matricial
- Red neuronal de 8 pasos (1000x20x20 → 1000x10)

