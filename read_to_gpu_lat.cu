//nvcc read_to_gpu_lat.cu     -I/home/duo/.local/include     -L/home/duo/.local/lib/x86_64-linux-gnu     -lcxl_shm     -Xlinker -rpath     -Xlinker /home/duo/.local/lib/x86_64-linux-gnu     -o read_to_gpu_lat     -O2 -g

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>

extern "C" {
#include <cxl_shm/cacheline.h>
#include <cxl_shm/api.h>
}

#define DEFAULT_NUM_OBJECTS 20

/*
 * 20.48 MB, decimal:
 * 20.48 × 1,000,000 = 20,480,000 bytes
 */
#define DEFAULT_OBJECT_SIZE ((size_t)800000000)

/*
 * 类似 perftest 的 warm-up。
 * warm-up 不计入最终 latency。
 */
#define DEFAULT_WARMUP 10

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

static inline uint64_t get_time_ns(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (uint64_t)ts.tv_sec * 1000000000ULL +
           (uint64_t)ts.tv_nsec;
}

static void make_key(char *key, size_t key_size, size_t index)
{
    snprintf(key, key_size, "object_%04zu", index);
}

static int compare_double(const void *a, const void *b)
{
    const double da = *(const double *)a;
    const double db = *(const double *)b;

    if (da < db) {
        return -1;
    }

    if (da > db) {
        return 1;
    }

    return 0;
}

static double percentile(const double *sorted,
                         size_t count,
                         double fraction)
{
    if (count == 0) {
        return 0.0;
    }

    double position = fraction * (double)(count - 1);
    size_t lower = (size_t)position;
    size_t upper = lower + 1;

    if (upper >= count) {
        return sorted[count - 1];
    }

    double weight = position - (double)lower;

    return sorted[lower] * (1.0 - weight) +
           sorted[upper] * weight;
}

static void *wait_for_object(const char *key)
{
    void *buffer = NULL;

    while (cxl_shm_get(key, &buffer) != 0) {
        usleep(100);
    }

    return buffer;
}

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s [object_size_bytes] [iterations] [warmup]\n"
            "\n"
            "Examples:\n"
            "  %s\n"
            "  %s 20480000 1000 10\n"
            "  %s 800000000 100 5\n",
            program,
            program,
            program,
            program);
}

int main(int argc, char **argv)
{
    size_t object_size = DEFAULT_OBJECT_SIZE;
    size_t iterations = DEFAULT_NUM_OBJECTS;
    size_t warmup = DEFAULT_WARMUP;
    const int gpu_id = 0;

    if (argc > 1) {
        object_size = strtoull(argv[1], NULL, 10);
    }

    if (argc > 2) {
        iterations = strtoull(argv[2], NULL, 10);
    }

    if (argc > 3) {
        warmup = strtoull(argv[3], NULL, 10);
    }

    if (argc > 4 ||
        object_size == 0 ||
        iterations == 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (cxl_shm_init() != 0) {
        fprintf(stderr, "cxl_shm_init failed\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp device_property;
    CUDA_CHECK(cudaGetDeviceProperties(&device_property, gpu_id));

    printf("GPU              : %s\n", device_property.name);
    printf("GPU index        : %d\n", gpu_id);
    printf("Object size      : %zu bytes\n", object_size);
    printf("Iterations       : %zu\n", iterations);
    printf("Warm-up          : %zu\n", warmup);
    printf("Outstanding copy : 1\n");

    /*
     * GPU 上只需要一个 object-sized buffer。
     * 每次 object 都复制到同一块 GPU buffer。
     */
    unsigned char *gpu_buffer = NULL;
    CUDA_CHECK(cudaMalloc((void **)&gpu_buffer, object_size));

    /*
     * Non-blocking stream，便于明确地对单个 copy 做同步。
     */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking));

    /*
     * 确保 CUDA context 初始化完成，不把首次初始化开销计入测试。
     */
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     * 先找到所有 object 的 CXL 地址。
     *
     * 这样正式 latency 可以只测 CXL -> GPU 数据传输，
     * 不包含 key lookup。
     *
     * 如果你想包含 cxl_shm_get()，后面有修改方法。
     */
    void **cxl_buffers =
        (void **)calloc(iterations, sizeof(void *));

    if (cxl_buffers == NULL) {
        fprintf(stderr, "calloc for CXL pointers failed\n");
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(gpu_buffer));
        return EXIT_FAILURE;
    }

    printf("Resolving CXL objects...\n");

    for (size_t i = 0; i < iterations; i++) {
        char key[64];

        make_key(key, sizeof(key), i);
        cxl_buffers[i] = wait_for_object(key);

        if (cxl_buffers[i] == NULL) {
            fprintf(stderr,
                    "NULL CXL pointer for key %s\n",
                    key);
            free(cxl_buffers);
            CUDA_CHECK(cudaStreamDestroy(stream));
            CUDA_CHECK(cudaFree(gpu_buffer));
            return EXIT_FAILURE;
        }
    }

    printf("All CXL objects resolved.\n");

    /*
     * Warm-up。
     * 如果 warmup 大于 object 数量，则循环使用已有 objects。
     */
    printf("Running warm-up...\n");

    for (size_t i = 0; i < warmup; i++) {
        size_t object_index = i % iterations;

        CUDA_CHECK(cudaMemcpyAsync(
            gpu_buffer,
            cxl_buffers[object_index],
            object_size,
            cudaMemcpyHostToDevice,
            stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    double *latencies_us =
        (double *)malloc(iterations * sizeof(double));

    if (latencies_us == NULL) {
        fprintf(stderr, "malloc for latency array failed\n");
        free(cxl_buffers);
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(gpu_buffer));
        return EXIT_FAILURE;
    }

    printf("Starting latency test...\n");

    uint64_t total_start_ns = get_time_ns();

    for (size_t i = 0; i < iterations; i++) {
        /*
         * 前一轮已经同步结束，因此任意时刻只存在一个
         * outstanding CXL -> GPU copy。
         */
        uint64_t start_ns = get_time_ns();

        CUDA_CHECK(cudaMemcpyAsync(
            gpu_buffer,
            cxl_buffers[i],
            object_size,
            cudaMemcpyHostToDevice,
            stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        uint64_t end_ns = get_time_ns();

        latencies_us[i] =
            (double)(end_ns - start_ns) / 1000.0;
    }

    uint64_t total_end_ns = get_time_ns();

    /*
     * 排序用于 median 和 percentile。
     */
    qsort(latencies_us,
          iterations,
          sizeof(double),
          compare_double);

    double sum_us = 0.0;

    for (size_t i = 0; i < iterations; i++) {
        sum_us += latencies_us[i];
    }

    const double min_us = latencies_us[0];
    const double max_us = latencies_us[iterations - 1];
    const double avg_us = sum_us / (double)iterations;
    const double median_us =
        percentile(latencies_us, iterations, 0.50);
    const double p95_us =
        percentile(latencies_us, iterations, 0.95);
    const double p99_us =
        percentile(latencies_us, iterations, 0.99);

    double total_seconds =
        (double)(total_end_ns - total_start_ns) /
        1000000000.0;

    double total_bytes =
        (double)object_size * (double)iterations;

    double bandwidth_gb_s =
        total_bytes / total_seconds / 1.0e9;

    double bandwidth_gbit_s =
        bandwidth_gb_s * 8.0;

    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
    printf(" #bytes      #iterations   min[usec]   max[usec]   median[usec]   avg[usec]   p99[usec]\n");
    printf(" %zu   %zu        %.3f      %.3f       %.3f          %.3f      %.3f\n",
           object_size,
           iterations,
           min_us,
           max_us,
           median_us,
           avg_us,
           p99_us);
    printf("--------------------------------------------------------------------------------\n");

    printf("P95 latency       : %.3f us\n", p95_us);
    printf("Total elapsed     : %.6f s\n", total_seconds);
    printf("Effective BW      : %.3f GB/s\n", bandwidth_gb_s);
    printf("Effective BW      : %.3f Gb/s\n", bandwidth_gbit_s);
    printf("Average latency   : %.6f ms/object\n",
           avg_us / 1000.0);

    free(latencies_us);
    free(cxl_buffers);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(gpu_buffer));

    return EXIT_SUCCESS;
}