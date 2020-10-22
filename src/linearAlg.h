#ifndef LINEAR_ALG_H
#define LINEAR_ALG_H

#define TILE_WIDTH 32
#define TILE_DIM 32

#define BLOCK_SIZE 32
#include <math.h>
#include <stdio.h>

void matrixMultiplyHost(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns);
int matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns);
__global__ void matrixMultiplyKernel(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns);

void matrixMultiply(int * A, int * B, int * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns);
__global__ void matrixMultiplyKernel(int * A, int * B, int * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns);

//////
void sigmoida(float *A, float *B, int rows, int cols);
__global__ void sigmoidaKernel(float *A, float *B, int rows, int cols);


///////
void transpose(int *A, int *B, int rows, int cols);
void transposeHost(float *A, float *B, int rows, int cols);
int transpose(float *A, float *B, int rows, int cols);
__global__ void transposeKernel(float *idata, float *odata, int height, int width);
__global__ void transposeMatrixFast(float * inputMatrix, float * outputMatrix, int w, int h);
__global__ void transposeMatrixFast(int * inputMatrix, int * outputMatrix, int w, int h);



//////
void subMat(float *A, float *B, float*C, int rows, int cols);
__global__ void subKernelMat(float *A, float *B, float *C, int rows, int cols);

/////
void addMat(float *A, float *B, float*C, int rows, int cols);
__global__ void addKernelMat(float *A, float *B, float *C, int rows, int cols);

////
void mulMat(float *A, float *B, float*C, int rows, int cols);
__global__ void mulKernelMat(float *A, float *B, float *C, int rows, int cols);


///
void subNum(float *A, float num, float*C, int rows, int cols);
__global__ void subKernelNum(float *A, float num, float *C, int rows, int cols);

///
void addNum(float *A, float num, float*C, int rows, int cols);
__global__ void addKernelNum(float *A, float num, float *C, int rows, int cols);


///
void mulNum(float *A, float num, float*C, int rows, int cols);
__global__ void mulKernelNum(float *A, float num, float *C, int rows, int cols);

















#endif
