#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#define BLOCK_SIZE 32

void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = static_cast<float>(rand() % 100);
        }
    }
}

bool checkMarticesEquality(const float* a, const float* b, int N) {
    bool areEqual = true;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (a[i * N + j] != b[i * N + j]) {
                areEqual = false;
                i = N;
                j = N;
            }
        }
    }

    return areEqual;
}

void matrixMultiplyCPU(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

__global__ void kernel_global(float* a, float* b, int N, float* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int ia = N * (BLOCK_SIZE * by + ty);
    int ib = BLOCK_SIZE * bx + tx;      
    int ic = ia + ib;  

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += a[ia + k] * b[ib + k * N];
    }
    c[ic] = sum;
}

__global__ void kernel_smem_1(float* a, float* b, int N, float* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Индексы для доступа к данным
    int aBegin = N * BLOCK_SIZE * by; 
    int aEnd = aBegin + N - 1;     
    int bBegin = BLOCK_SIZE * bx; 
    int aStep = BLOCK_SIZE;        
    int bStep = BLOCK_SIZE * N; 

    float sum = 0.0f;

    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        as[tx][ty] = a[ia + N * ty + tx]; 
        bs[tx][ty] = b[ib + N * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += as[k][ty] * bs[tx][k];
        }

        __syncthreads();
    }

    c[aBegin + bBegin + ty * N + tx] = sum;
}

__global__ void kernel_smem_2(float* a, float* b, int N, float* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Индексы для доступа к данным
    int aBegin = N * BLOCK_SIZE * by;
    int aEnd = aBegin + N - 1;
    int bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE;   
    int bStep = BLOCK_SIZE * N;   

    float sum = 0.0f;

    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        as[tx][ty] = a[ia + N * ty + tx];
        bs[tx][ty] = b[ib + N * ty + tx];

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += as[k][ty] * bs[tx][k];
        }

        __syncthreads();
    }

    c[aBegin + bBegin + ty * N + tx] = sum;
}

__global__ void kernel_smem_3(float* a, float* b, int N, float* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int aBegin = N * BLOCK_SIZE * by;
    int aEnd = aBegin + N - 1;
    int bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE;
    int bStep = BLOCK_SIZE * N;

    float sum = 0.0f;

    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        as[ty][tx] = a[ia + N * ty + tx];
        bs[ty][tx] = b[ib + N * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += as[ty][k] * bs[k][tx];
        }
        __syncthreads();
    }

    c[aBegin + bBegin + ty * N + tx] = sum;
}

__global__ void kernel_smem_4(float* a, float* b, int N, float* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int aBegin = N * BLOCK_SIZE * by;
    int aEnd = aBegin + N - 1;
    int bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE;
    int bStep = BLOCK_SIZE * N;

    float sum1 = 0.0f, sum2 = 0.0f;

    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        as[ty][tx] = a[ia + N * ty + tx];
        bs[ty][tx] = b[ib + N * ty + tx];
        as[ty + BLOCK_SIZE / 2][tx] = a[ia + N * (ty + BLOCK_SIZE / 2) + tx];
        bs[ty + BLOCK_SIZE / 2][tx] = b[ib + N * (ty + BLOCK_SIZE / 2) + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum1 += as[ty][k] * bs[k][tx];
            sum2 += as[ty + BLOCK_SIZE / 2][k] * bs[k][tx];
        }
        __syncthreads();
    }

    c[aBegin + bBegin + ty * N + tx] = sum1;
    c[aBegin + bBegin + (ty + BLOCK_SIZE / 2) * N + tx] = sum2;
}

__global__ void kernel_smem_5(float* a, float* b, int N, float* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int aBegin = N * BLOCK_SIZE * by;
    int aEnd = aBegin + N - 1;
    int bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE;
    int bStep = BLOCK_SIZE * N;

    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        as[ty][tx] = a[ia + N * ty + tx];
        bs[ty][tx] = b[ib + N * ty + tx];
        as[ty + BLOCK_SIZE / 4][tx] = a[ia + N * (ty + BLOCK_SIZE / 4) + tx];
        bs[ty + BLOCK_SIZE / 4][tx] = b[ib + N * (ty + BLOCK_SIZE / 4) + tx];
        as[ty + BLOCK_SIZE / 2][tx] = a[ia + N * (ty + BLOCK_SIZE / 2) + tx];
        bs[ty + BLOCK_SIZE / 2][tx] = b[ib + N * (ty + BLOCK_SIZE / 2) + tx];
        as[ty + 3 * BLOCK_SIZE / 4][tx] = a[ia + N * (ty + 3 * BLOCK_SIZE / 4) + tx];
        bs[ty + 3 * BLOCK_SIZE / 4][tx] = b[ib + N * (ty + 3 * BLOCK_SIZE / 4) + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum1 += as[ty][k] * bs[k][tx];
            sum2 += as[ty + BLOCK_SIZE / 4][k] * bs[k][tx];
            sum3 += as[ty + BLOCK_SIZE / 2][k] * bs[k][tx];
            sum4 += as[ty + 3 * BLOCK_SIZE / 4][k] * bs[k][tx];
        }
        __syncthreads();
    }

    c[aBegin + bBegin + ty * N + tx] = sum1;
    c[aBegin + bBegin + (ty + BLOCK_SIZE / 4) * N + tx] = sum2;
    c[aBegin + bBegin + (ty + BLOCK_SIZE / 2) * N + tx] = sum3;
    c[aBegin + bBegin + (ty + 3 * BLOCK_SIZE / 4) * N + tx] = sum4;
}


int main() {
    const int N = 2048;
    const int SIZE = N * N * sizeof(float);

    float* a, * b, * cCPU, * cGPU, * cSMEM1, * cSMEM2, * cSMEM3, * cSMEM4, * cSMEM5;
    a = (float*)malloc(SIZE);
    b = (float*)malloc(SIZE);
    cCPU = (float*)malloc(SIZE);
    cGPU = (float*)malloc(SIZE);
    cSMEM1 = (float*)malloc(SIZE);
    cSMEM2 = (float*)malloc(SIZE);
    cSMEM3 = (float*)malloc(SIZE);
    cSMEM4 = (float*)malloc(SIZE);
    cSMEM5 = (float*)malloc(SIZE);

    initializeMatrix(a, N);
    initializeMatrix(b, N);

    float* aDev, * bDev, * cDev;
    cudaMalloc((void**)&aDev, SIZE);
    cudaMalloc((void**)&bDev, SIZE);
    cudaMalloc((void**)&cDev, SIZE);

    cudaMemcpy(aDev, a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(bDev, b, SIZE, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --------------------- CPU ---------------------
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(a, b, cCPU, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpuTime = endCPU - startCPU;
    std::cout << "CPU Calculation Time: " << cpuTime.count() << " ms\n";

    // --------------------- GPU Global Memory ---------------------
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 threads_smem4(BLOCK_SIZE, BLOCK_SIZE / 2);
    dim3 threads_smem5(BLOCK_SIZE, BLOCK_SIZE / 4);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    cudaEventRecord(start, 0);
    
    kernel_global << <blocks, threads >> > (aDev, bDev, N, cDev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuGlobalTime;
    cudaEventElapsedTime(&gpuGlobalTime, start, stop);
    cudaMemcpy(cGPU, cDev, SIZE, cudaMemcpyDeviceToHost);

    std::cout << "GPU Global Memory Time: " << gpuGlobalTime << " ms\n";
    if (checkMarticesEquality(cCPU, cGPU, N)) std::cout << "GPU = CPU\n";
    
    // --------------------- GPU SMEM-1 ---------------------
    cudaEventRecord(start, 0);
    
    kernel_smem_1 << <blocks, threads >> > (aDev, bDev, N, cDev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuSMEM1Time;
    cudaEventElapsedTime(&gpuSMEM1Time, start, stop);
    std::cout << "GPU SMEM-1 Time: " << gpuSMEM1Time << " ms\n";
    cudaMemcpy(cSMEM1, cDev, SIZE, cudaMemcpyDeviceToHost);
    if (checkMarticesEquality(cCPU, cSMEM1, N)) std::cout << "SMEM1 = CPU\n";

    // --------------------- GPU SMEM-2 ---------------------
    cudaEventRecord(start, 0);
    
    kernel_smem_2 << <blocks, threads >> > (aDev, bDev, N, cDev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuSMEM2Time;
    cudaEventElapsedTime(&gpuSMEM2Time, start, stop);
    std::cout << "GPU SMEM-2 Time: " << gpuSMEM2Time << " ms\n";
    cudaMemcpy(cSMEM2, cDev, SIZE, cudaMemcpyDeviceToHost);
    if (checkMarticesEquality(cCPU, cSMEM2, N)) std::cout << "SMEM2 = CPU\n";

    // --------------------- GPU SMEM-3 ---------------------
    cudaEventRecord(start, 0);
    
    kernel_smem_3 << <blocks, threads >> > (aDev, bDev, N, cDev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuSMEM3Time;
    cudaEventElapsedTime(&gpuSMEM3Time, start, stop);
    std::cout << "GPU SMEM-3 Time: " << gpuSMEM3Time << " ms\n";
    cudaMemcpy(cSMEM3, cDev, SIZE, cudaMemcpyDeviceToHost);
    if (checkMarticesEquality(cCPU, cSMEM3, N)) std::cout << "SMEM3 = CPU\n";

    // --------------------- GPU SMEM-4 ---------------------
    cudaEventRecord(start, 0);
    
    kernel_smem_4 << <blocks, threads_smem4 >> > (aDev, bDev, N, cDev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuSMEM4Time;
    cudaEventElapsedTime(&gpuSMEM4Time, start, stop);
    std::cout << "GPU SMEM-4 Time: " << gpuSMEM4Time << " ms\n";
    cudaMemcpy(cSMEM4, cDev, SIZE, cudaMemcpyDeviceToHost);
    if (checkMarticesEquality(cCPU, cSMEM4, N)) std::cout << "SMEM4 = CPU\n";

    // --------------------- GPU SMEM-5 ---------------------
    cudaEventRecord(start, 0);
    
    kernel_smem_5 << <blocks, threads_smem5 >> > (aDev, bDev, N, cDev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuSMEM5Time;
    cudaEventElapsedTime(&gpuSMEM5Time, start, stop);
    std::cout << "GPU SMEM-5 Time: " << gpuSMEM5Time << " ms\n";
    cudaMemcpy(cSMEM5, cDev, SIZE, cudaMemcpyDeviceToHost);
    if (checkMarticesEquality(cCPU, cSMEM5, N)) std::cout << "SMEM5 = CPU\n";

    double timeOfCPU = cpuTime.count();

    std::cout << "Speedup (Global): " << timeOfCPU / gpuGlobalTime << "x\n";
    std::cout << "Speedup (SMEM-1): " << timeOfCPU / gpuSMEM1Time << "x\n";
    std::cout << "Speedup (SMEM-2): " << timeOfCPU / gpuSMEM2Time << "x\n";
    std::cout << "Speedup (SMEM-3): " << timeOfCPU / gpuSMEM3Time << "x\n";
    std::cout << "Speedup (SMEM-4): " << timeOfCPU / gpuSMEM4Time << "x\n";
    std::cout << "Speedup (SMEM-5): " << timeOfCPU / gpuSMEM5Time << "x\n";

    cudaFree(aDev);
    cudaFree(bDev);
    cudaFree(cDev);
    free(a);
    free(b);
    free(cCPU);
    free(cGPU);
    free(cSMEM1);
    free(cSMEM2);
    free(cSMEM3);
    free(cSMEM4);
    free(cSMEM5);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
