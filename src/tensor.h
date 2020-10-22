#include <iostream>
#include <map>
#include "linearAlg.h"


class tensor{
public:
    tensor(float* data, int* size, int dim, bool autograd, int id = -1);

    tensor(float* data, int* size, int dim, int id = -1);

    tensor(float* data, int* size, int dim, std::string creation_op, tensor* right, tensor* left, bool autograd);

    // tensor(float* data);

    tensor(tensor *src);

    tensor();

    tensor(tensor &t);

    tensor& operator= ( tensor &t);

    ~tensor();

    float operator[](int i);

    bool all_children_grads_accounted_for();

    int* getSize();

    int getVolume();

    int getDim();

    void print();

    void backward(tensor* grad, tensor* grad_origin = NULL);

// private:
    tensor* grad = NULL;
    bool autograd = false;
    float* data = NULL;
    int* size = NULL;
    int id = -1;
    int dim;
    int volume;
    std::string creation_op;
    tensor* left = NULL;
    tensor* right = NULL;
    std::map<int, int> children;
};

tensor* add (tensor* f, tensor*  t);

tensor* neg (tensor* f);

tensor* sub (tensor* f, tensor*  t);

tensor* mul (tensor* f, tensor*  t);

tensor* sum (tensor* f, int dim);

tensor* expand (tensor* f, int dim, int copies);

tensor* transpose (tensor* f);

tensor* mm (tensor* f,  tensor*  t);
