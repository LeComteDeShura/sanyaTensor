
// #include "tensor.h"
#include <iostream>
#include <map>

void add(float* a, float* b, float* c, int size){
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}


class tensor{
public:
    tensor(float* data, int* size, int dim, bool autograd, int id = -1){
        this->data = data;
        this->size = size;
        this->dim = dim;
        this->volume = 1;
        this->autograd = autograd;
        for (int i = 0; i < dim; i++) {
            this->volume *= size[i];
        }
        if (id == -1) {
            this->id = 1 + rand() % 100000;
        }
        this->id = id;
    }

    tensor(float* data, int* size, int dim, std::string creation_op, tensor* right, tensor* left, bool autograd){
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
        this->autograd = autograd;
        if (id == -1) {
            id = 1 + rand() % 100000;
        }

        if (left != NULL && right != NULL) {
            auto temp = left->children.find(this->id);
            if (temp == left->children.end()) {
                left->children.insert(std::make_pair(id, 1));
            }
            else{
                left->children[id] += 1;
            }

            temp = right->children.find(this->id);
            if (temp == right->children.end()) {
                right->children.insert(std::make_pair(this->id, 1));
            }
            else{
                right->children[this->id] += 1;
            }
        }
    }

    float operator[](int i){
        return data[i];
    }

    bool all_children_grads_accounted_for(){
        auto temp = this->children.begin();
        while (temp != children.end()) {
            if (temp->second != 0) {
                return false;
            }
            temp++;
        }
        return true;
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

    void backward(float* grad, tensor* grad_origin = NULL){
        if (autograd) {
            if (grad_origin != NULL) {
                // std::cout << children[grad_origin->id] << '\n';
                if (children[grad_origin->id] == 0) {
                    std::cout << "cannot backprop more than once" << '\n';
                    return;
                }
                else{
                    children[grad_origin->id] -= 1;
                }
            }
            if (this->grad == NULL) {
                this->grad = grad;
            }
            else{
                add(this->grad, grad, this->grad, volume);
            }
            // if (grad_origin != NULL) {
            //     std::cout << children[grad_origin->id] << '\n';
            //     std::cout << all_children_grads_accounted_for() << '\n';
            // }
            if (left != NULL && right != NULL && (all_children_grads_accounted_for() || grad_origin == NULL)  ) {
                if (creation_op == "add") {
                    // std::cout << "2/* message */" << '\n';
                    left->backward(grad, this);
                    right->backward(grad, this);
                }
            }
        }
    }

// private:
    float* grad = NULL;
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

tensor* add (tensor* f, tensor*  t){
    if (f->getVolume() != t->getVolume() && f->getSize()[0] != t->getSize()[0]) {
        // throw "Can not take square root of negative number";
        throw std::invalid_argument("Dim +.");
    }

    float* aret = new float[t->getVolume()];

    for (int i = 0; i < f->getVolume(); i++) {
        aret[i] = (*t)[i] + (*f)[i];
    }
    if (f->autograd && t->autograd) {
        tensor* tret = new tensor(aret, f->getSize(), t->getDim(), "add", t, f, true);
        // std::cout << f->id << '\n';
        return tret;
    }
    tensor* tret = new tensor(aret, f->getSize(), t->getDim(), "add", t, f, false);
    return tret;
}

void print(float* array, int size){
    if (array == NULL) {
        std::cout << "error" << '\n';
        return;
    }
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
    // int size2[] = {2,4};
    float arr[] =  {1,1,1,1,1,1};
    float arra[] = {2,2,2,3,3,3};
    float arrb[] = {2,2,2,3,3,3};
    float arre[] = {2,2,2,3,3,3};
    // float arrd[] = {4,5,6,7,8,9};
    int dim = 2;
    tensor* a = new tensor(arra, size, dim, true, 1);
    tensor* b = new tensor(arrb, size, dim, true, 2);
    tensor* c = new tensor(arre, size, dim, true, 3);
    // tensor* d = new tensor(arrd, size, dim);

    tensor *d = add(a, b);
    tensor *e = add(b, c);
    tensor *f = add(d, e);
    // f->print();
    f->backward(arr);
    // std::cout << "2123" << '\n';
    print(b->grad, 6);

    // std::map<int, int> test;
    // test.insert(std::make_pair(1, 1));
    // test[1] += 1;
    // std::cout << test[2] << '\n';

    return 0;
}







//
