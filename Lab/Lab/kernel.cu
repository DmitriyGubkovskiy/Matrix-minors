
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <time.h> 
#include <iostream>

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#define BASE char

#define CHECK_CUDA_ERROR(call)                                      \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            /* Можно вывести название ошибки: cudaGetErrorName(err) */ \
            /* Можно завершить программу или обработать иначе */    \
            exit(1);                                                \
        }                                                           \
    } while (0)


const int matrixSize = 100;
__constant__ BASE matrixDev[matrixSize * matrixSize];

void generateMatrix(BASE * matrix, int size);
__device__ void createMinor(BASE * matrix, int* size, short* result, int* x, int* y);
__device__ void createMinor(int* size, short* result, int* x, int* y);
__global__ void kernel(BASE * matrix, int* size, short* minor);
__global__ void kernel(int* size, short* minor);


int main()
{ 
    std::srand(time(NULL));
    // conditions 
    const int numbersOfBlocks[4]{10, 100, 1000, 10000};
    const int numbersOfThreads[1]{1000};
    const short sizeOfMatrix[1]{32};
    /*const int sizeOfMatrix[1]{ 100 };
    const int numbersOfBlocks[1]{ 10000};
    const int numbersOfThreads[1]{1000};*/

    /*int countBlocks = 1;
    int countThreads = 10;*/

    for (int countBlocks : numbersOfBlocks)
    {
        for (int countThreads : numbersOfThreads)
        {
            /*if (countBlocks == 10 && countThreads > 10) continue;
            if (countBlocks > 10 && countThreads > 1) continue;*/
            // data for host
            BASE* matrix = new BASE[matrixSize * matrixSize];
            short* minors = new short[matrixSize * matrixSize];

            // data for device
            /*BASE* matrixDev;*/

            
            int* matrixSizeDev;
            short* minorsDev;

               
            generateMatrix(matrix, matrixSize);
    
            /*for (int i = 0; i < matrixSize; i++)
            {
                for (int j = 0; j < matrixSize; j++)
                {
                    printf("%d ", matrix[j + i * matrixSize]);
                }
                printf("\n");
            }*/

            /*int* matrix = new int[16] {10, 20, 30, 4, 5, 6, 7, 8, 9,10,110,12,13,14,15,16};*/
            /*BASE* matrix = new BASE[matrixSize * matrixSize] {5, 7, 1, -4, 1, 0, 2, 0, 3};*/


            /*for (int i = 0; i < matrixSize; i++)
            {
                for (int j = 0; j < matrixSize; j++)
                {
                    printf("%d ", matrix[j + i * matrixSize]);
                }
                printf("\n");
            }*/
            clock_t start = clock();

            /*CHECK_CUDA_ERROR(cudaMalloc((void**)&matrixDev, sizeof(BASE) * matrixSize * matrixSize));*/
            CHECK_CUDA_ERROR(cudaMalloc((void**)&matrixSizeDev, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&minorsDev, sizeof(short) * matrixSize * matrixSize));

            /*CHECK_CUDA_ERROR(cudaMemcpy(matrixDev, matrix, sizeof(BASE) * matrixSize * matrixSize, cudaMemcpyHostToDevice));*/
            CHECK_CUDA_ERROR(cudaMemcpy(matrixSizeDev, &matrixSize, sizeof(int), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(matrixDev, matrix, sizeof(BASE) * matrixSize * matrixSize));
            

            kernel << <countBlocks, countThreads >> > (matrixSizeDev, minorsDev);
            /*kernel << <countBlocks, countThreads >> > (matrixDev, matrixSizeDev, minorsDev);*/
            CHECK_CUDA_ERROR(cudaGetLastError());   // Проверка ошибок запуска ядра
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            CHECK_CUDA_ERROR(cudaMemcpy(minors, minorsDev, sizeof(short) * matrixSize * matrixSize, cudaMemcpyDeviceToHost));


            /*for (int i = 0; i < matrixSize; i++)
            {
                for (int j = 0; j < matrixSize; j++)
                {
                    printf("%d ", minors[j + i * matrixSize]);
                }
                printf("\n");
            }*/

            /*cudaFree(matrixDev);*/
            cudaFree(matrixSizeDev);
            cudaFree(minorsDev);

            delete[] matrix;
            delete[] minors;

            clock_t end = clock();
            double seconds = (double)(end - start) / CLOCKS_PER_SEC * 1000;

            printf("Runtime %f ms for %d processors and %d threads with matrix size %d. \n", 
                seconds, countBlocks, countThreads, matrixSize);
        }
    }
    return 0;
}

void generateMatrix(BASE* matrix, int size) {


    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[j + i * size] = (std::rand() % 2 + 1);
        }
    }

}

__device__ void createMinor(BASE* matrix, int* size, short* result, int* x, int* y) {
    BASE det = 1;
    int n = *size;

    BASE* m = new BASE[(n - 1) * (n - 1)];

    int indX = 0;

    for (int i = 0; i < n; i++)
    {
        int indY = 0;
        if (i == (*x)) continue;
        for (int j = 0; j < n; j++)
        {
            if (j == (*y)) continue;
            m[indY + indX * (n - 1)] = matrix[j + i * n];
            indY++;
        }
        indX++;
    }

    n--;

    for (int i = 0; i < n; ++i)
    {
        BASE mx = abs(m[i + i * n]);
        int idx = i;
        for (int j = i + 1; j < n; ++j)
            if (mx < abs(m[j + i * n])) {
                idx = j;
                mx = abs(m[j + i * n]);
            }
        if (idx != i)
        {
            for (int j = i; j < n; ++j)
            {
                BASE t = m[i + j * n];
                m[i + j * n] = m[idx + j * n];
                m[idx + j * n] = t;
            }
            det = -det;
        }
        for (int k = i + 1; k < n; ++k)
        {
            BASE t = m[i + k * n] / (m[i + i * n] == 0 ? 1 : m[i + i * n]);

            for (int j = i; j < n; ++j)
                m[j + k * n] -= m[j + i * n] * t;
        }
    }
    for (int i = 0; i < n; ++i) det *= m[i + i * n];
    result[(*y) + (*x) * (n + 1)] = det;
    delete[] m;

}

__device__ void createMinor(int* size, short* result, int* x, int* y) {
    BASE det = 1;
    int n = *size;

    BASE* m = new BASE[(n - 1) * (n - 1)];

    int indX = 0;

    for (int i = 0; i < n; i++)
    {
        int indY = 0;
        if (i == (*x)) continue;
        for (int j = 0; j < n; j++)
        {
            if (j == (*y)) continue;
            m[indY + indX * (n - 1)] = matrixDev[j + i * n];
            indY++;
        }
        indX++;
    }

    n--;

    for (int i = 0; i < n; ++i)
    {
        BASE mx = abs(m[i + i * n]);
        int idx = i;
        for (int j = i + 1; j < n; ++j)
            if (mx < abs(m[j + i * n])) {
                idx = j;
                mx = abs(m[j + i * n]);
            }
        if (idx != i)
        {
            for (int j = i; j < n; ++j)
            {
                BASE t = m[i + j * n];
                m[i + j * n] = m[idx + j * n];
                m[idx + j * n] = t;
            }
            det = -det;
        }
        for (int k = i + 1; k < n; ++k)
        {
            BASE t = m[i + k * n] / (m[i + i * n] == 0 ? 1 : m[i + i * n]);

            for (int j = i; j < n; ++j)
                m[j + k * n] -= m[j + i * n] * t;
        }
    }
    for (int i = 0; i < n; ++i) det *= m[i + i * n];
    result[(*y) + (*x) * (n + 1)] = det;
    delete[] m;

}


__global__ void kernel(BASE* matrix, int* size, short* minor) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int n = (*size);
    if (threadId >= n * n) return;


    while (threadId < n * n) {

        int column = threadId % n;
        int row = threadId / n;
        createMinor(matrix, size, minor, &row, &column);
        threadId += blockDim.x * gridDim.x;
    }
}

__global__ void kernel(int* size, short* minor) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int n = (*size);
    if (threadId >= n * n) return;


    while (threadId < n * n) {

        int column = threadId % n;
        int row = threadId / n;
        createMinor(size, minor, &row, &column);
        threadId += blockDim.x * gridDim.x;
    }
}


