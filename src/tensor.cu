#include "tensor.h"


tensor::tensor(float* data, int* size, int dim, bool autograd, int id){
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

tensor::tensor(float* data, int* size, int dim, int id){
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

tensor::tensor(float* data, int* size, int dim, std::string creation_op, tensor* right, tensor* left, bool autograd){
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

float tensor::operator[](int i){
    return data[i];
}

bool tensor::all_children_grads_accounted_for(){
    auto temp = this->children.begin();
    while (temp != children.end()) {
        if (temp->second != 0) {
            return false;
        }
        temp++;
    }
    return true;
}

int* tensor::getSize(){
    return size;
}

int tensor::getVolume(){
    return volume;
}

int tensor::getDim(){
    return dim;
}

void tensor::print(){
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

void tensor::backward(tensor* grad, tensor* grad_origin){
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
            // add(this->grad, grad, this->grad, volume);
            this->grad = add(this->grad, grad);
        }
        if (left != NULL && right != NULL && (all_children_grads_accounted_for() || grad_origin == NULL)  ) {
            if (creation_op == "add") {
                // std::cout << "2/* message */" << '\n';
                left->backward(grad, this);
                right->backward(grad, this);
            }
        }
    }
}

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











//
