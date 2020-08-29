
#include "tensor.h"

class tensor{
public:
    tensor(float* data, int* size, int dim){
        this->data = data;
        this->size = size;
        this->dim = dim;
        this->volume = 1;
        for (int i = 0; i < dim; i++) {
            this->volume *= size[i];
        }
    }

    tensor(float* data, int* size, int dim, std::string creation_op, tensor* right, tensor* left){
        this->data = data;
        this->size = size;
        this->dim = dim;
        this->volume = 1;
        for (int i = 0; i < dim; i++) {
            this->volume *= size[i];
        }
        this->right = right;
        this->left = left;
        this->creation_op = creation_op;
    }

    float operator[](int i){
        return data[i];
    }

    int* getSize(){
        return size;
    }

    int getVolume(){
        return volume;
    }

    int getDim(){
        return dim;
    }

    void print(){
        if (dim == 2) {
            for (int i = 0; i < size[0]; i++) {
                std::cout << "[";
                for (int j = 0; j < size[1]; j++) {
                    std::cout << std::fixed;
                    std::cout.precision(2);
                    std::cout << data[i * size[1] + j];
                    if (j != size[1] -1 ) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]\n";
            }
        }
        else{
            std::cout << "[";
            for (int i = 0; i < volume; i++) {
                std::cout << data[i];
                if (i != volume -1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]\n";
        }
    }

    void backward(float* grad){
        this->grad = grad;
        if (creation_op == "add") {
            left->backward(grad);
            right->backward(grad);
        }
    }

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

tensor* add (tensor* f, tensor*  t){
    if (f->getVolume() != t->getVolume() && f->getSize()[0] != t->getSize()[0]) {
        throw "Can not take square root of negative number";
    }

    float* aret = new float[t->getVolume()];

    for (int i = 0; i < f->getVolume(); i++) {
        aret[i] = (*t)[i] + (*f)[i];
    }
    tensor* tret = new tensor(aret, f->getSize(), t->getDim(), "add", t, f);
    return tret;
}

void print(float* array, int size){
    std::cout << "(";
    for (int i = 0; i < size; i++) {
        std::cout << array[i];
        if (size != i + 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";
}

int main(int argc, char const *argv[]) {
    int size[] = {2,3};
    float arra[] = {1,2,3,1,2,3};
    float arrb[] = {5,4,3,2,1,1};
    float arre[] = {9,8,7,6,5,4};
    float arrd[] = {4,5,6,7,8,9};
    int dim = 2;
    tensor* a = new tensor(arra, size, dim);
    tensor* b = new tensor(arrb, size, dim);
    tensor* c = new tensor(arre, size, dim);
    tensor* d = new tensor(arrd, size, dim);

    tensor *e = add(a, b);
    tensor *f = add(c, d);
    tensor *g = add(e, f);
    g->backward(arra);
    print(a->grad, 6);
    g->print();
    return 0;
}







//
