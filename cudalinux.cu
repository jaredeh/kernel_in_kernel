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
    struct fakefolio *fakefolios_host;
    struct fakefolio *fakefolios;
    unsigned long *output_host;
    unsigned long *output;


    fakefolio_init(&fakefolios_host, &output_host);
    // Allocate – accessible from GPU
    cudaMalloc(&output,sizeof(unsigned long)*TOTAL_ENTRIES);
    cudaMalloc(&fakefolios,sizeof(struct fakefolio)*TOTAL_ENTRIES);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "malloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(fakefolios, fakefolios_host, sizeof(struct fakefolio)*TOTAL_ENTRIES, cudaMemcpyHostToDevice);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "memcpy1 failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    std::cout << "initialized fakefolios" << std::endl;

    // Launch kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (TOTAL_ENTRIES+threadsPerBlock-1)/threadsPerBlock;

    cudaDeviceSynchronize();
 
    cudaEventRecord(start);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaKernel prelaunch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(fakefolios, output);
    // Check for any errors from kernel launch
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaMemcpy(output_host, output, sizeof(unsigned long)*TOTAL_ENTRIES, cudaMemcpyDeviceToHost);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

     printf("GPU Duration: %f ms\n", milliseconds);

    // Write the output to a binary file
    FILE *fp = fopen("cudaoutput.bin","wb");
    fwrite(output_host, sizeof(unsigned long), TOTAL_ENTRIES, fp);
    fclose(fp);

    // Free memory
    cudaFree(fakefolios);
    cudaFree(output);

    return 0;
}
