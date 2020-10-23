#include "tensor.h"

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
    else{
        this->id = id;
    }
}

tensor::tensor(float* data, int* size, int dim, int id){
    this->data = data;
    this->size = size;
    this->dim = dim;
    this->volume = 1;
    this->autograd = false;
    for (int i = 0; i < dim; i++) {
        this->volume *= size[i];
    }
    if (id == -1) {
        this->id = 1 + rand() % 100000;
    }
    else{
        this->id = id;
    }
}

tensor::tensor(){}

tensor::tensor(tensor *src){
    this->volume = src->getVolume();
    this->data = new float(this->volume);
    this->data = (float*)memccpy(this->data, src->data, '*', this->volume);
    if (this->data == NULL) {
        throw std::invalid_argument("Failed to free array data");
    }

    this->dim = src->getDim();
    this->size = new int(this->dim);
    this->size = (int*)memccpy(this->size, src->size, '*', this->dim);
    if (this->size == NULL) {
        throw std::invalid_argument("Failed to free array size");
    }

    this->id = src->id;
    this->autograd = src->autograd;

    this->left = src->left;
    this->right = src->right;

    this->creation_op = src->creation_op;
    this->grad = src->grad;
    this->children = src->children;
}

tensor::tensor(tensor &src){
    if (this == &src) {
        return;
    }
    this->volume = src.getVolume();
    this->data = new float(this->volume);
    this->data = (float*)memccpy(this->data, src.data, '*', this->volume);
    if (this->data == NULL) {
        throw std::invalid_argument("Failed to free array data");
    }

    this->dim = src.getDim();
    this->size = new int(this->dim);
    this->size = (int*)memccpy(this->size, src.size, '*', this->dim);
    if (this->size == NULL) {
        throw std::invalid_argument("Failed to free array size");
    }

    this->id = src.id;
    this->autograd = src.autograd;

    this->left = src.left;
    this->right = src.right;

    this->creation_op = src.creation_op;
    this->grad = src.grad;
    this->children = src.children;
}

tensor& tensor::operator= (tensor &src){
    this->volume = src.getVolume();
    this->data = new float(this->volume);
    this->data = (float*)memccpy(this->data, src.data, '*', this->volume);
    if (this->data == NULL) {
        throw std::invalid_argument("Failed to free array data");
    }

    this->dim = src.getDim();
    this->size = new int(this->dim);
    this->size = (int*)memccpy(this->size, src.size, '*', this->dim);
    if (this->size == NULL) {
        throw std::invalid_argument("Failed to free array size");
    }

    this->id = src.id;
    this->autograd = src.autograd;

    this->left = src.left;
    this->right = src.right;

    this->creation_op = src.creation_op;
    this->grad = src.grad;
    this->children = src.children;
    return *this;
}

tensor::~tensor(){
    delete [] data;
    delete [] size;
    delete left;
    delete right;
    delete grad;
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
    if(dim == 1){
        std::cout << "[";
        for (int i = 0; i < volume; i++) {
            std::cout << data[i];
            if (i != volume -1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    if (dim == 3) {
        std::cout << "[";
        for (int n = 0; n < size[2]; n++) {
            for (int i = 0; i < size[0]; i++) {
                if (i == 0 && n == 0) {
                    std::cout << "[";
                }
                else{
                    std::cout << " [";
                }
                for (int j = 0; j < size[1]; j++) {
                    std::cout << std::fixed;
                    std::cout.precision(2);
                    std::cout << data[n * size[0] * size[1] + i * size[1] + j];
                    if (j != size[1] -1 ) {
                        std::cout << ", ";
                    }
                }
                if (i != size[0] - 1) {
                    std::cout << "],\n";
                }
            }

            if (n != size[2]-1) {
                std::cout << "];\n\n";
                // std::cout << '\n';
            }
            else{
                std::cout << "]";
            }
        }
        std::cout << "]";
        std::cout  << '\n';
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
        if ( (left != NULL || right != NULL) && (all_children_grads_accounted_for() || grad_origin == NULL)  ) {
            if (creation_op == "add") {
                // std::cout << "2/* message */" << '\n';
                left->backward(this->grad, this);
                right->backward(this->grad, this);
            }

            if (creation_op == "neg") {
                right->backward(neg(this->grad));
            }

            if (creation_op == "sub") {
                // int v = this->grad->getVolume();
                // float* nw1 = new float[v];
                // nw1 = (float*)std::memcpy((void*)nw1, (void*)this->grad->data, v);
                //
                // int d = this->grad->getDim();
                //
                // int* s1 = new int[d];
                // s1 = (int*)std::memcpy((void*)s1, (void*)this->grad->getSize(), d);

                // left->backward(new tensor(nw1, s1, v, d), this);
                left->backward(new tensor(this->grad), this);
                // ///
                // float* nw2 = new float[v];
                // nw2 = (float*)std::memcpy((void*)nw2, (void*)this->grad->data, v);
                //
                // int* s2 = new int[d];
                // s2 = (int*)std::memcpy((void*)s2, (void*)this->grad->getSize(), d);

                // right->backward(neg(new tensor(nw2, s2, v, d)), this);
                right->backward(neg(new tensor(this->grad)), this);
            }

            if (creation_op == "mul") {
                tensor* nw1 = mul(this->grad, this->right);
                left->backward(nw1, this);
                tensor* nw2 = mul(this->grad, this->left);
                right->backward(nw2, this);
            }

            if (creation_op == "mm") {
                tensor* nw1 = mm(this->grad, this->right);
                left->backward(nw1);
                tensor* nw2 = transpose(mul(transpose(this->grad), this->left));
                right->backward(nw2);
            }

            if (creation_op == "transpose") {
                this->right->backward(transpose(this->grad));
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

tensor* neg (tensor* f){
    float* aret = new float[f->getVolume()];

    for (int i = 0; i < f->getVolume(); i++) {
        aret[i] = (*f)[i] * -1;
    }
    if (f->autograd) {
        tensor* tret = new tensor(aret, f->getSize(), f->getDim(), "neg", f, NULL, true);
        return tret;
    }
    tensor* tret = new tensor(aret, f->getSize(), f->getDim(), "neg", f, NULL, false);
    return tret;
}

tensor* sub (tensor* f, tensor*  t){
    if (f->getVolume() != t->getVolume() && f->getSize()[0] != t->getSize()[0]) {
        // throw "Can not take square root of negative number";
        throw std::invalid_argument("Dim +.");
    }

    float* aret = new float[t->getVolume()];

    for (int i = 0; i < f->getVolume(); i++) {
        aret[i] = (*f)[i] - (*t)[i];
    }
    if (f->autograd && t->autograd) {
        tensor* tret = new tensor(aret, f->getSize(), t->getDim(), "sub", t, f, true);
        return tret;
    }
    tensor* tret = new tensor(aret, f->getSize(), t->getDim(), "sub", t, f, false);
    return tret;
}

tensor* mul (tensor* f, tensor*  t){
    if (f->getVolume() != t->getVolume() && f->getSize()[0] != t->getSize()[0]) {
        // throw "Can not take square root of negative number";
        throw std::invalid_argument("Dim +.");
    }

    float* aret = new float[t->getVolume()];

    for (int i = 0; i < f->getVolume(); i++) {
        aret[i] = (*f)[i] * (*t)[i];
    }
    if (f->autograd && t->autograd) {
        tensor* tret = new tensor(aret, f->getSize(), t->getDim(), "mul", t, f, true);
        return tret;
    }
    tensor* tret = new tensor(aret, f->getSize(), t->getDim(), "mul", t, f, false);
    return tret;
}

tensor* sum (tensor* f, int dim){
    if (dim > f->dim) {
        std::cout << "error sum" << '\n';
    }
    // dim = 2;
    int volume = 1;
    int volume_area = 1;

    int *size = new int[f->dim - 1];
    int l = 0;
    for (int i = 0; i < f->dim; i++) {
        size[i - l] = f->size[i];
        if (i == dim-1) {
            l++;
        }
    }

    for (int i = 0; i < f->dim; i++) {
        volume_area *= f->size[i];
    }
    float* ar = new float[volume_area];
    volume = volume_area / f->size[dim-1];


    int step;
    int s;
    int area;
    if (dim == 2) {
        step = 1;
        s = f->size[1];
        area = volume_area;
    }

    else if (dim == 1) {
        s = 1;
        area = volume;
        step = volume;
    }

    int n  = 0;
    for (int i = 0; i < area; i += s) {
        for (int j = 0; j < f->size[dim-1]; j ++) {
            ar[n] += (*f)[step * j + i];
            std::cout << (*f)[step * j + i] << " + ";
        }
        n++;
        std::cout  << '\n';
    }



    if (f->autograd) {
        tensor* tret = new tensor(ar, size, f->getDim() - 1, "sum", f, NULL, true);
        return tret;
    }
    tensor* tret = new tensor(ar, size, f->getDim() - 1, "sum", f, NULL, false);
    return tret;
}

tensor* expand (tensor* f, int dim, int copies){
    float* ar = new float;
    int *size = new int[f->dim + 1];
    if(dim == 1){
        ar = new float[f->volume * copies];

        for (int i = 0; i < copies; i++) {
            for (int j = 0; j < f->volume; j++) {
                ar[i * f->volume + j] = f->data[j];
            }
        }

        int l = 0;
        for (int i = 0; i < f->dim + 1; i++) {
            if (i == 2) {
                size[i] = copies;
                l = 1;
                continue;
            }
            size[i] = f->size[i - l];
        }
    }

    if(dim == 2){
        ar = new float[f->volume * copies];
        for (int i = 0; i < f->volume; i++) {
            for (int j = 0; j < copies; j++) {
                ar[i * copies + j] = f->data[i];
            }
        }

        int l = 0;
        for (int i = 0; i < f->dim + 1; i++) {
            if (i == 1) {
                size[i] = copies;
                l = 1;
                continue;
            }
            size[i] = f->size[i - l];
        }
    }

    if (f->autograd) {
        tensor* tret = new tensor(ar, size, f->getDim() + 1, "expand", f, NULL, true);
        return tret;
    }
    tensor* tret = new tensor(ar, size, f->getDim() + 1, "expand", f, NULL, false);
    return tret;
}

tensor* mm (tensor* f, tensor*  t){
    if (f->getSize()[1] != t->getSize()[0]) {
        throw std::invalid_argument("Dim +.");
    }

    int *size = new int[f->getDim()];
    size[0] = f->getSize()[0];
    size[1] = t->getSize()[1];
    int sizek = t->getSize()[0];

    float* aret = new float[size[0] * size[1]];

    matrixMultiply(f->data, t->data, aret, size[0], sizek, sizek, size[1], size[0], size[1]);


    if (f->autograd && t->autograd) {
        tensor* tret = new tensor(aret, size, t->getDim(), "mm", f, t, true);
        return tret;
    }
    tensor* tret = new tensor(aret, size, t->getDim(), "mm", f, t, false);
    return tret;
}

tensor* transpose (tensor* f){
    int *size = new int[f->getDim()];
    size[0] = f->getSize()[0];
    size[1] = f->getSize()[1];

    float* aret = new float[size[0] * size[1]];

    transpose(f->data, aret, size[0], size[1]);


    if (f->autograd) {
        tensor* tret = new tensor(aret, size, f->getDim(), "transpose", f, NULL, true);
        return tret;
    }
    tensor* tret = new tensor(aret, size, f->getDim(), "transpose", f, NULL, false);
    return tret;
}










//
