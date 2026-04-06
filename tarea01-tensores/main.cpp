#include "TensorTransform.h"
#include "Tensor.h"
#include <iostream>


//g++ main.cpp Tensor.cpp -o main
//cd "c:\Users\Usuario\tarea1Progra3\tarea01-tensores\" ; if ($?) { g++ main.cpp Tensor.cpp -o main } ; if ($?) { .\main }
int main(){
    Tensor A = Tensor::zeros({3,3});
    Tensor B = Tensor::ones({3,3});
    Tensor C = Tensor::random({2,2}, 0.0, 1.0);
    Tensor D = Tensor::arange(0,6);

    Tensor E = A + B;
    Tensor F = A - B;
    Tensor G = A * B;
    Tensor H = A * 2.0;

    ReLU relu;
    Sigmoid sig;

 // Seccion 7: view
    Tensor D_view = D.view({2, 3});
    cout << "view ok: " << D_view.getShape()[0] << "x" << D_view.getShape()[1] << endl;

    // Seccion 7: unsqueeze
    Tensor vec = Tensor::arange(0, 3);
    Tensor vec_us0 = vec.unsqueeze(0); // {1, 3}
    Tensor vec_us1 = vec.unsqueeze(1); // {3, 1}
    cout << "unsqueeze(0): " << vec_us0.getShape()[0] << "x" << vec_us0.getShape()[1] << endl;
    cout << "unsqueeze(1): " << vec_us1.getShape()[0] << "x" << vec_us1.getShape()[1] << endl;

    // Seccion 8: concat
    Tensor T1 = Tensor::ones({2, 3});
    Tensor T2 = Tensor::zeros({2, 3});
    Tensor T3 = Tensor::concat({T1, T2}, 0); // {4, 3}
    cout << "concat shape: " << T3.getShape()[0] << "x" << T3.getShape()[1] << endl;

    // Seccion 9: dot
    Tensor v1 = Tensor::arange(0, 4);
    Tensor v2 = Tensor::ones({4});
    Tensor resultado_dot = dot(v1, v2);
    cout << "dot([0,1,2,3], [1,1,1,1]) = " << resultado_dot.getData()[0] << endl;

    // Seccion 9: matmul
    Tensor M1 = Tensor::arange(0, 6).view({2, 3});
    Tensor M2 = Tensor::ones({3, 2});
    Tensor MM = matmul(M1, M2);
    cout << "matmul shape: " << MM.getShape()[0] << "x" << MM.getShape()[1] << endl;

    // ============================================
    // Seccion 10: Red Neuronal
    // ============================================
    cout << "\n--- Red Neuronal ---" << endl;

    // Paso 1: entrada 1000x20x20
    Tensor entrada = Tensor::random({1000, 20, 20}, 0.0, 1.0);
    cout << "Paso 1. Se crea un tensor de entrada: " << entrada.getShape()[0] << "x"
         << entrada.getShape()[1] << "" << entrada.getShape()[2] << endl;

    // Paso 2: view a 1000x400
    Tensor entrada_flat = entrada.view({1000, 400});
    cout << "Paso 2. Transformando a: " << entrada_flat.getShape()[0] << "x" << entrada_flat.getShape()[1] << endl;

    // Paso 3: matmul con W1 (400x100) => resultado 1000x100
    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor capa1 = matmul(entrada_flat, W1);
    cout << "Paso 3.Se multiplica por una matriz de 400x100: " << capa1.getShape()[0] << "x" << capa1.getShape()[1] << endl;

    // Paso 4: suma bias b1 (1000x100, simplificado como ceros)
    Tensor b1 = Tensor::zeros({1000, 100});
    Tensor capa1_bias = capa1 + b1;
    cout << "Paso 4: bias resultante  " << capa1_bias.getShape()[0] << "x" << capa1_bias.getShape()[1] << endl;

    // Paso 5: ReLU
    Tensor capa1_relu = capa1_bias.apply(relu);
    cout << "Paso 5: activacion de capa ReLU" << capa1_relu.getShape()[0] << "x" << capa1_relu.getShape()[1] << endl;

    // Paso 6: matmul con W2 (100x10) => resultado 1000x10
    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor capa2 = matmul(capa1_relu, W2);
    cout << "Paso 6: matmul con pesos W2 " << capa2.getShape()[0] << "x" << capa2.getShape()[1] << endl;

    // Paso 7: suma bias b2 (1000x10, simplificado como ceros)
    Tensor b2 = Tensor::zeros({1000, 10});
    Tensor capa2_bias = capa2 + b2;
    cout << "Paso 7: bias sumado ok   " <<  capa2_bias.getShape()[0] << "x" << capa2_bias.getShape()[1] << endl;

    // Paso 8: Sigmoid
    Tensor salida = capa2_bias.apply(sig);
    cout << "Paso 8: Sigmoid ok  " <<  capa2_bias.getShape()[0] << "x" << capa2_bias.getShape()[1] << endl;

    cout << "\nShape final: " << salida.getShape()[0] << "x" << salida.getShape()[1] << endl;


   
    return 0;


}