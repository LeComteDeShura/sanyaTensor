#include <iostream>

lass tensor{
public:
    tensor(float* data, int* size, int dim);
    tensor(float* data, int* size, int dim, std::string creation_op, tensor* right, tensor* left);

    float operator[](int i);

    int* getSize();
    int getVolume();
    int getDim()

    void print();
    void backward(float* grad);

private:
    float* data = NULL;
    int* size = NULL;
    float* grad = NULL;
    int dim;
    int volume;
    std::string creation_op;
    tensor* left = NULL;
    tensor* right = NULL;

};
