#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <thread> // Include for std::this_thread::sleep_for
#include <chrono>
#include "fakefolio.h"


// CUDA Kernel function to initialize the array elements
__global__ void cudaKernel(struct fakefolio *fakefolios, unsigned long *output, int size) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    output[idx] = fakefolios[idx].mapcount;
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
    int threadsPerBlock = 128;
    int blocksPerGrid = (TOTAL_ENTRIES+threadsPerBlock-1)/threadsPerBlock;
    cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(fakefolios, output);


    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();


    // Write the output to a binary file
    FILE *fp = fopen("cudaoutput.bin","wb");
    fwrite(output, sizeof(unsigned long), TOTAL_ENTRIES, fp);
    fclose(fp);

    // Free memory
    cudaFree(fakefolios);
    cudaFree(output);

    return 0;
}
