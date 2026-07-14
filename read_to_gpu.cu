#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

extern "C" {
#include <cxl_shm/cacheline.h>
#include <cxl_shm/api.h>
}

#define NUM_OBJECTS 20
#define OBJECT_SIZE ((size_t)800000000)

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr,                                                \
                    "CUDA error at %s:%d: %s\n",                            \
                    __FILE__,                                              \
                    __LINE__,                                              \
                    cudaGetErrorString(err__));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

static void make_key(char *key, size_t key_size, size_t index)
{
    snprintf(key, key_size, "object_%04zu", index);
}

static void wait_for_object(const char *key, void **buffer)
{
    while (cxl_shm_get(key, buffer) != 0) {
        usleep(1000);
    }
}

int main(void)
{
    const int gpu_id = 0;

    const size_t total_size =
        (size_t)NUM_OBJECTS * (size_t)OBJECT_SIZE;

    if (cxl_shm_init() != 0) {
        fprintf(stderr, "cxl_shm_init failed\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp property;
    CUDA_CHECK(cudaGetDeviceProperties(&property, gpu_id));

    printf("GPU           : %s\n", property.name);
    printf("Object count  : %d\n", NUM_OBJECTS);
    printf("Object size   : %zu bytes\n", OBJECT_SIZE);
    printf("Total size    : %.3f GiB\n",
           (double)total_size /
           (1024.0 * 1024.0 * 1024.0));

    size_t free_gpu_memory = 0;
    size_t total_gpu_memory = 0;

    CUDA_CHECK(cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory));

    printf("GPU free      : %.3f GiB\n",
           (double)free_gpu_memory /
           (1024.0 * 1024.0 * 1024.0));

    printf("GPU total     : %.3f GiB\n",
           (double)total_gpu_memory /
           (1024.0 * 1024.0 * 1024.0));

    if (free_gpu_memory < total_size) {
        fprintf(stderr,
                "Not enough GPU memory: need %.3f GiB, free %.3f GiB\n",
                (double)total_size /
                    (1024.0 * 1024.0 * 1024.0),
                (double)free_gpu_memory /
                    (1024.0 * 1024.0 * 1024.0));
        return EXIT_FAILURE;
    }

    unsigned char *gpu_buffer = NULL;

    printf("Allocating GPU buffer...\n");
    fflush(stdout);

    CUDA_CHECK(cudaMalloc((void **)&gpu_buffer, total_size));

    printf("GPU buffer address: %p\n", gpu_buffer);
    fflush(stdout);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event));

    for (size_t i = 0; i < NUM_OBJECTS; i++) {
        char key[64];
        void *cxl_buffer = NULL;

        make_key(key, sizeof(key), i);
        wait_for_object(key, &cxl_buffer);

        if (cxl_buffer == NULL) {
            fprintf(stderr, "NULL pointer returned for key %s\n", key);
            CUDA_CHECK(cudaFree(gpu_buffer));
            return EXIT_FAILURE;
        }

        unsigned char *gpu_destination =
            gpu_buffer + i * OBJECT_SIZE;

        /*
         * 将 CXL CPU-visible memory 复制到 GPU device memory。
         */
        CUDA_CHECK(cudaMemcpy(gpu_destination,
                              cxl_buffer,
                              OBJECT_SIZE,
                              cudaMemcpyHostToDevice));

        if ((i + 1) % 10 == 0 || i == 0) {
            printf("Copied %zu/%d objects, %.3f GiB\n",
                   i + 1,
                   NUM_OBJECTS,
                   ((double)(i + 1) * OBJECT_SIZE) /
                   (1024.0 * 1024.0 * 1024.0));
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,
                                    start_event,
                                    stop_event));

    double elapsed_seconds = elapsed_ms / 1000.0;
    double bandwidth_gb_s =
        ((double)total_size / 1.0e9) / elapsed_seconds;

    printf("\nCopy completed\n");
    printf("Elapsed time  : %.6f seconds\n", elapsed_seconds);
    printf("Bandwidth     : %.3f GB/s\n", bandwidth_gb_s);

    /*
     * 验证每个 object 开头保存的 uint64_t index。
     * 只从 GPU 拷回 8 bytes/object，不需要把全部 20 GB 拷回来。
     */
    printf("Verifying object headers...\n");

    for (size_t i = 0; i < NUM_OBJECTS; i++) {
        uint64_t value = UINT64_MAX;

        CUDA_CHECK(cudaMemcpy(
            &value,
            gpu_buffer + i * OBJECT_SIZE,
            sizeof(value),
            cudaMemcpyDeviceToHost));

        if (value != (uint64_t)i) {
            fprintf(stderr,
                    "Verification failed for object %zu: "
                    "expected %zu, got %llu\n",
                    i,
                    i,
                    (unsigned long long)value);

            CUDA_CHECK(cudaFree(gpu_buffer));
            return EXIT_FAILURE;
        }
    }

    printf("Verification passed for all %d objects.\n",
           NUM_OBJECTS);

    printf("Keeping GPU allocation alive. Press Ctrl+C to stop.\n");
    fflush(stdout);

    for (;;) {
        pause();
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(gpu_buffer));

    return EXIT_SUCCESS;
}