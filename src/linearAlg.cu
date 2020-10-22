#include "linearAlg.h"


void matrixMultiplyHost(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
    int threadsPerBlockDim = TILE_WIDTH;
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim, 1); // Grid: X x Y x Z=1

    int blocksPerGridDimX = ceilf(numCColumns / (float)threadsPerBlockDim);
    int blocksPerGridDimY = ceilf(numCRows / (float)threadsPerBlockDim);
    dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);

    matrixMultiplyKernel<<<gridDim, blockDim>>>(A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaDeviceSynchronize();
}

int matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc((void **)&d_a, numARows * numAColumns*sizeof(float));
    cudaMalloc((void **)&d_b, numBRows * numBColumns*sizeof(float));
    cudaMalloc((void **)&d_c, numCRows * numCColumns*sizeof(float));

    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(d_a, A, numARows * numAColumns*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_b, B, numBRows * numBColumns*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    matrixMultiplyHost(d_a, d_b, d_c, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    err = cudaMemcpy(C, d_c , numARows * numBColumns*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
__global__ void matrixMultiplyKernel(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

void matrixMultiply(int * A, int * B, int * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
    int threadsPerBlockDim = TILE_WIDTH;
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim, 1); // Grid: X x Y x Z=1

    int blocksPerGridDimX = ceilf(numCColumns / (float)threadsPerBlockDim);
    int blocksPerGridDimY = ceilf(numCRows / (float)threadsPerBlockDim);
    dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);

    matrixMultiplyKernel<<<gridDim, blockDim>>>(A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    // printf("\n%d\n", C[0]);
    cudaDeviceSynchronize();
}
__global__ void matrixMultiplyKernel(int * A, int * B, int * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    int Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

//////
void sigmoida(float *A, float *B, int rows, int cols){
    int threadsPerBlockDim = TILE_WIDTH;
    int blocksPerGridDim = ceilf(rows*cols / (float)threadsPerBlockDim);
    sigmoidaKernel<<<blocksPerGridDim, threadsPerBlockDim>>>(A, B, rows, cols);

    cudaDeviceSynchronize();
}
__global__ void sigmoidaKernel(float *A, float *B, int rows, int cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float e = 2.7180;
    for (int i = index; i < rows*cols; i += stride)
        B[i] = 1 / (1 + pow(e, (A[i] * -1)));
}


///////
void transpose(int *A, int *B, int rows, int cols){
    int threadsPerBlockDim = TILE_WIDTH;
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim, 1); // Grid: X x Y x Z=1

    int blocksPerGridDimX = ceilf(cols / (float)threadsPerBlockDim);
    int blocksPerGridDimY = ceilf(rows / (float)threadsPerBlockDim);
    dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);
    transposeMatrixFast<<<gridDim, blockDim>>>(A, B, cols, rows);

    cudaDeviceSynchronize();
}
void transposeHost(float *A, float *B, int rows, int cols){
    int threadsPerBlockDim = TILE_WIDTH;
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim, 1); // Grid: X x Y x Z=1

    int blocksPerGridDimX = ceilf(cols / (float)threadsPerBlockDim);
    int blocksPerGridDimY = ceilf(rows / (float)threadsPerBlockDim);
    dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);
    transposeMatrixFast<<<gridDim, blockDim>>>(A, B, cols, rows);

    cudaDeviceSynchronize();
}

int transpose(float *A, float *B, int rows, int cols){
    float* d_a;
    float* d_b;

    cudaMalloc((void **)&d_a, rows * cols*sizeof(float));
    cudaMalloc((void **)&d_b, rows * cols*sizeof(float));

    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(d_a, A, rows * cols*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    transposeHost(d_a, d_b, rows, cols);

    err = cudaMemcpy(B, d_b , rows * cols*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

__global__ void transposeKernel(float *idata, float *odata, int height, int width){
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	tile[threadIdx.y][threadIdx.x] = idata[index_in];
	__syncthreads();
	odata[index_out] = tile[threadIdx.x][threadIdx.y];
}
__global__ void transposeMatrixFast(int * inputMatrix, int * outputMatrix, int w, int h){
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ((xIndex < w) && (yIndex < h)){
        int idx = yIndex * w + xIndex;
        temp[threadIdx.y][threadIdx.x] = inputMatrix[idx];
    }
    __syncthreads();
    xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    if ((xIndex < h) && (yIndex < w)){
        int idx = yIndex * h + xIndex;
        outputMatrix[idx] = temp[threadIdx.x][threadIdx.y];
    }
}
__global__ void transposeMatrixFast(float * inputMatrix, float * outputMatrix, int w, int h){
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ((xIndex < w) && (yIndex < h)){
        int idx = yIndex * w + xIndex;
        temp[threadIdx.y][threadIdx.x] = inputMatrix[idx];
    }
    __syncthreads();
    xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    if ((xIndex < h) && (yIndex < w)){
        int idx = yIndex * h + xIndex;
        outputMatrix[idx] = temp[threadIdx.x][threadIdx.y];
    }
}

//////
void subMat(float *A, float *B, float*C, int rows, int cols){
    int threadsPerBlockDim = TILE_DIM;
    int blocksPerGridDim = ceilf(rows*cols / (float)threadsPerBlockDim);
    subKernelMat<<<blocksPerGridDim, threadsPerBlockDim>>>(A, B, C, rows, cols);

    cudaDeviceSynchronize();
}
__global__ void subKernelMat(float *A, float *B, float *C, int rows, int cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < rows*cols; i += stride)
        C[i] = A[i] - B[i];
}

/////
void addMat(float *A, float *B, float*C, int rows, int cols){
    int threadsPerBlockDim = TILE_DIM;
    int blocksPerGridDim = ceilf(rows*cols / (float)threadsPerBlockDim);
    addKernelMat<<<blocksPerGridDim, threadsPerBlockDim>>>(A, B, C, rows, cols);

    cudaDeviceSynchronize();
}
__global__ void addKernelMat(float *A, float *B, float *C, int rows, int cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = index; i < rows*cols; i += stride)
        C[i] = A[i] + B[i];
}

////
void mulMat(float *A, float *B, float*C, int rows, int cols){
    int threadsPerBlockDim = TILE_DIM;
    int blocksPerGridDim = ceilf(rows*cols / (float)threadsPerBlockDim);
    mulKernelMat<<<blocksPerGridDim, threadsPerBlockDim>>>(A, B, C, rows, cols);

    cudaDeviceSynchronize();
}
__global__ void mulKernelMat(float *A, float *B, float *C, int rows, int cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = index; i < rows*cols; i += stride)
        C[i] = A[i] * B[i];
}


///
void subNum(float *A, float num, float*C, int rows, int cols){
    int threadsPerBlockDim = TILE_DIM;
    int blocksPerGridDim = ceilf(rows*cols / (float)threadsPerBlockDim);
    subKernelNum<<<blocksPerGridDim, threadsPerBlockDim>>>(A, num, C, rows, cols);

    cudaDeviceSynchronize();
}
__global__ void subKernelNum(float *A, float num, float *C, int rows, int cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < rows*cols; i += stride)
        C[i] = num - A[i];
}

///
void addNum(float *A, float num, float*C, int rows, int cols){
    int threadsPerBlockDim = TILE_DIM;
    int blocksPerGridDim = ceilf(rows*cols / (float)threadsPerBlockDim);
    addKernelNum<<<blocksPerGridDim, threadsPerBlockDim>>>(A, num, C, rows, cols);

    cudaDeviceSynchronize();
}
__global__ void addKernelNum(float *A, float num, float *C, int rows, int cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < rows*cols; i += stride)
        C[i] = num + A[i];
}


///
void mulNum(float *A, float num, float*C, int rows, int cols){
    int threadsPerBlockDim = TILE_DIM;
    int blocksPerGridDim = ceilf(rows*cols / (float)threadsPerBlockDim);
    mulKernelNum<<<blocksPerGridDim, threadsPerBlockDim>>>(A, num, C, rows, cols);

    cudaDeviceSynchronize();
}
__global__ void mulKernelNum(float *A, float num, float *C, int rows, int cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < rows*cols; i += stride)
        C[i] = num * A[i];
}























//
