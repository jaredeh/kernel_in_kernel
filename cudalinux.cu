#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <thread> // Include for std::this_thread::sleep_for
#include <chrono>
#include "fakefolio.h"
#include <sys/time.h>


// CUDA Kernel function to initialize the array elements
__global__ void cudaKernel(struct fakefolio *fakefolios, unsigned long *output) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    output[idx] = fakefolios[idx].mapcount;
}

unsigned long long dtime_usec(unsigned long long start=0) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((tv.tv_sec * 1000000ULL) + tv.tv_usec) - start;
}

// // CUDA Kernel function to initialize the array elements
// __global__ void cudaKernel(struct fakefolio *fakefolios, unsigned long *output, int size) {
//     int idx = blockIdxthreadIdx.x*size;
//     for (int i = 0; i < size; i++) {
//         output[i+idx] = fakefolios[i+idx].mapcount;
//     }   
// }


int main() {
    struct fakefolio *fakefolios;
    unsigned long *output;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&output,sizeof(unsigned long)*TOTAL_ENTRIES);
    cudaMallocManaged(&fakefolios,sizeof(struct fakefolio)*TOTAL_ENTRIES);
    for (int i = 0; i < TOTAL_ENTRIES; i++) {
        fakefolios[i].mapcount = i;
    }
    std::cout << "initialized fakefolios" << std::endl;


    // Launch kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (TOTAL_ENTRIES+threadsPerBlock-1)/threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(fakefolios, output);

     cudaEventRecord(stop);


    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize(stop);

    / Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Convert milliseconds to microseconds
    float microseconds = milliseconds * 1000;
    printf("GPU Duration: %f us\n", microseconds);


    // Write the output to a binary file
    FILE *fp = fopen("cudaoutput.bin","wb");
    fwrite(output, sizeof(unsigned long), TOTAL_ENTRIES, fp);
    fclose(fp);

    // Free memory
    cudaFree(fakefolios);
    cudaFree(output);

    return 0;
}
