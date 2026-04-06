#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void fill_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2;
    }
}

static void check(cudaError_t err, const char *step) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", step, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    cudaDeviceProp prop {};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("Using GPU 0: %s\n", prop.name);
    std::printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    std::printf("Global memory: %.2f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    constexpr int n = 256;
    int *device_data = nullptr;
    int host_data[n] = {};

    check(cudaMalloc(&device_data, n * sizeof(int)), "cudaMalloc");
    fill_kernel<<<1, n>>>(device_data, n);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaMemcpy(host_data, device_data, n * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check(cudaFree(device_data), "cudaFree");

    long long sum = 0;
    for (int value : host_data) {
        sum += value;
    }

    std::printf("Kernel output sum: %lld\n", sum);
    std::printf("CUDA smoke test passed.\n");
    return 0;
}
