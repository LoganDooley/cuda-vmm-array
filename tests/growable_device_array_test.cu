#include "cuda_vmm_array/growable_device_array.h"

#include <cstdio>
#include <vector>


// Macro for standard cuda error checking
#define CUDA_CHECK(call) \
    do { \
        const cudaError_t cudaCheckError = (call); \
        if (cudaCheckError != cudaSuccess) { \
            throw std::runtime_error(std::string(#call) + " failed at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " with error: " + \
                                     cudaGetErrorString(cudaCheckError)); \
        } \
    } while (0)

// Helper to random-ish data
__host__ __device__ inline uint32_t pattern(size_t i) {
    return uint32_t(i * 2654435761u) ^ 0x9E3779B9u;
}

__global__ void fillKernel(uint32_t* data, size_t begin, size_t end)
{
    const size_t i = begin + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < end) {
        data[i] = pattern(i);
    }
}

int main()
{
    try {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));

        // Reserve 256 MB of addressable space
        GrowableDeviceArray<uint32_t> array(
            0, size_t(64) << 20); 
        uint32_t* const firstAddress = nullptr;
        uint32_t* base = nullptr;
        size_t filled = 0;
        bool bPass = true;

        // Try sizes 1000, 600,000, 5,000,000, 20,000,000
        for (size_t wanted :
             {size_t(1000), size_t(600'000), size_t(5'000'000), size_t(20'000'000)}) {
            // Map memory
            array.ensureCapacity(wanted);

            if (base == nullptr) {
                // Cache the base pointer
                base = array.data();
            } else if (array.data() != base) {
                // This shoud not happen
                printf("FAIL: data() moved\n");
                bPass = false;
            }

            const size_t count = wanted - filled;
            const uint32_t threads = 256;
            // Fill new memory
            fillKernel<<<uint32_t((count + threads - 1) / threads), threads>>>(array.data(), filled,
                                                                               wanted);
            CUDA_CHECK(cudaGetLastError());
            filled = wanted;
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<uint32_t> host(filled);
            // Readback data to check
            CUDA_CHECK(cudaMemcpy(host.data(), array.data(), filled * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            size_t errors = 0;
            for (size_t i = 0; i < filled; i++) {
                // If any are not equal, errors > 0
                errors += host[i] != pattern(i);
            }
            printf("Wanted %10zu Capacity %10zu Mapped %7.1f MB Errors %zu\n", wanted,
                   array.capacity(), array.capacity() * sizeof(uint32_t) / 1048576.0, errors);
            bPass = bPass && errors == 0;
        }

        printf(bPass ? "PASS\n" : "FAIL\n");
        return bPass ? 0 : 1;
    } catch (const std::exception& e) {
        // Any exception is a fail
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}