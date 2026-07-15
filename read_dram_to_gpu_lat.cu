//nvcc read_dram_to_gpu_lat.cu     -I/home/duo/.local/include     -L/home/duo/.local/lib/x86_64-linux-gnu     -lcxl_shm     -Xlinker -rpath     -Xlinker /home/duo/.local/lib/x86_64-linux-gnu     -o dram_read_to_gpu_lat     -O2 -g
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

#define DEFAULT_GPU_ID        0
#define DEFAULT_NUM_OBJECTS   1000
#define DEFAULT_OBJECT_SIZE   ((size_t)20480000)
#define DEFAULT_WARMUP        10

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

static uint64_t get_time_ns(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (uint64_t)ts.tv_sec * 1000000000ULL +
           (uint64_t)ts.tv_nsec;
}

static int compare_double(const void *a, const void *b)
{
    double first = *(const double *)a;
    double second = *(const double *)b;

    if (first < second) {
        return -1;
    }

    if (first > second) {
        return 1;
    }

    return 0;
}

static double calculate_average(const double *values,
                                size_t count)
{
    if (count == 0) {
        return 0.0;
    }

    long double sum = 0.0;

    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }

    return (double)(sum / (long double)count);
}

static double calculate_sample_stdev(const double *values,
                                     size_t count,
                                     double average)
{
    if (count < 2) {
        return 0.0;
    }

    long double squared_sum = 0.0;

    for (size_t i = 0; i < count; i++) {
        long double difference =
            (long double)values[i] -
            (long double)average;

        squared_sum += difference * difference;
    }

    return sqrt((double)(
        squared_sum / (long double)(count - 1)));
}

static double calculate_median(const double *sorted_values,
                               size_t count)
{
    if (count == 0) {
        return 0.0;
    }

    if ((count % 2) == 1) {
        return sorted_values[count / 2];
    }

    return (
        sorted_values[count / 2 - 1] +
        sorted_values[count / 2]
    ) / 2.0;
}

static double percentile_nearest_rank(
    const double *sorted_values,
    size_t count,
    double percentile)
{
    if (count == 0) {
        return 0.0;
    }

    if (percentile <= 0.0) {
        return sorted_values[0];
    }

    if (percentile >= 100.0) {
        return sorted_values[count - 1];
    }

    size_t rank = (size_t)ceil(
        (percentile / 100.0) * (double)count);

    if (rank < 1) {
        rank = 1;
    }

    if (rank > count) {
        rank = count;
    }

    return sorted_values[rank - 1];
}

static void fill_object(void *buffer,
                        size_t object_size,
                        size_t object_index)
{
    unsigned char pattern =
        (unsigned char)(object_index % 251);

    memset(buffer, pattern, object_size);

    /*
     * 在每个 object 开头写入 object index，
     * 用于后续验证 GPU 中的数据。
     */
    if (object_size >= sizeof(uint64_t)) {
        uint64_t index_value =
            (uint64_t)object_index;

        memcpy(buffer,
               &index_value,
               sizeof(index_value));
    }
}

static void release_host_objects(void **objects,
                                 size_t allocated_count)
{
    if (objects == NULL) {
        return;
    }

    for (size_t i = 0; i < allocated_count; i++) {
        if (objects[i] != NULL) {
            cudaError_t err =
                cudaFreeHost(objects[i]);

            if (err != cudaSuccess) {
                fprintf(stderr,
                        "Warning: cudaFreeHost failed "
                        "for object %zu: %s\n",
                        i,
                        cudaGetErrorString(err));
            }
        }
    }

    free(objects);
}

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s [object_size_bytes] "
            "[iterations] [warmup] [gpu_id]\n"
            "\n"
            "Examples:\n"
            "  %s\n"
            "  %s 20480000 1000 10 0\n"
            "  %s 800000000 40 5 0\n",
            program,
            program,
            program,
            program);
}

int main(int argc, char **argv)
{
    size_t object_size =
        DEFAULT_OBJECT_SIZE;

    size_t iterations =
        DEFAULT_NUM_OBJECTS;

    size_t warmup_iterations =
        DEFAULT_WARMUP;

    int gpu_id =
        DEFAULT_GPU_ID;

    if (argc > 1) {
        object_size =
            (size_t)strtoull(argv[1], NULL, 10);
    }

    if (argc > 2) {
        iterations =
            (size_t)strtoull(argv[2], NULL, 10);
    }

    if (argc > 3) {
        warmup_iterations =
            (size_t)strtoull(argv[3], NULL, 10);
    }

    if (argc > 4) {
        gpu_id = atoi(argv[4]);
    }

    if (argc > 5 ||
        object_size == 0 ||
        iterations == 0 ||
        gpu_id < 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    /*
     * 检查 size_t 乘法是否溢出。
     */
    if (object_size > SIZE_MAX / iterations) {
        fprintf(stderr,
                "Total buffer size overflow: "
                "%zu x %zu\n",
                object_size,
                iterations);
        return EXIT_FAILURE;
    }

    size_t total_buffer_size =
        object_size * iterations;

    printf("Pinned DRAM -> GPU latency benchmark\n");
    printf("Object size           : %zu bytes\n",
           object_size);
    printf("Object size           : %.3f MB\n",
           (double)object_size / 1000000.0);
    printf("Number of objects     : %zu\n",
           iterations);
    printf("Warm-up iterations    : %zu\n",
           warmup_iterations);
    printf("Total dataset         : %.3f GB\n",
           (double)total_buffer_size / 1.0e9);
    printf("Total dataset         : %.3f GiB\n",
           (double)total_buffer_size /
           (1024.0 * 1024.0 * 1024.0));
    printf("Outstanding copies    : 1\n");
    printf("\n");

    int gpu_count = 0;

    CUDA_CHECK(cudaGetDeviceCount(&gpu_count));

    if (gpu_id >= gpu_count) {
        fprintf(stderr,
                "Invalid GPU ID %d. "
                "Available GPU count: %d\n",
                gpu_id,
                gpu_count);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));

    /*
     * 在正式测试前初始化 CUDA context。
     */
    CUDA_CHECK(cudaFree(0));

    cudaDeviceProp gpu_properties;

    CUDA_CHECK(cudaGetDeviceProperties(
        &gpu_properties,
        gpu_id));

    size_t free_gpu_memory = 0;
    size_t total_gpu_memory = 0;

    CUDA_CHECK(cudaMemGetInfo(
        &free_gpu_memory,
        &total_gpu_memory));

    printf("GPU ID                : %d\n",
           gpu_id);
    printf("GPU name              : %s\n",
           gpu_properties.name);
    printf("GPU total memory      : %.3f GiB\n",
           (double)total_gpu_memory /
           (1024.0 * 1024.0 * 1024.0));
    printf("GPU free memory       : %.3f GiB\n",
           (double)free_gpu_memory /
           (1024.0 * 1024.0 * 1024.0));
    printf("Required GPU memory   : %.3f GiB\n",
           (double)total_buffer_size /
           (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    /*
     * 留一点空间给 CUDA context 和其他 allocation。
     */
    const size_t safety_margin =
        (size_t)512 * 1024 * 1024;

    if (free_gpu_memory < total_buffer_size ||
        free_gpu_memory - total_buffer_size <
            safety_margin) {
        fprintf(stderr,
                "Not enough free GPU memory.\n");
        fprintf(stderr,
                "Required dataset : %.3f GiB\n",
                (double)total_buffer_size /
                (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr,
                "Free GPU memory  : %.3f GiB\n",
                (double)free_gpu_memory /
                (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr,
                "Safety margin    : %.3f GiB\n",
                (double)safety_margin /
                (1024.0 * 1024.0 * 1024.0));
        return EXIT_FAILURE;
    }

    /*
     * Host 端：为每个 object 单独分配 pinned DRAM。
     */
    void **dram_object_pointers =
        (void **)calloc(iterations,
                        sizeof(void *));

    if (dram_object_pointers == NULL) {
        fprintf(stderr,
                "Failed to allocate host pointer array\n");
        return EXIT_FAILURE;
    }

    printf("Allocating and initializing pinned DRAM objects...\n");

    size_t allocated_objects = 0;

    for (size_t i = 0; i < iterations; i++) {
        cudaError_t allocation_error =
            cudaHostAlloc(
                &dram_object_pointers[i],
                object_size,
                cudaHostAllocDefault);

        if (allocation_error != cudaSuccess) {
            fprintf(stderr,
                    "\ncudaHostAlloc failed at object %zu\n",
                    i);
            fprintf(stderr,
                    "CUDA error: %s\n",
                    cudaGetErrorString(allocation_error));
            fprintf(stderr,
                    "Pinned DRAM allocated before failure: "
                    "%.3f GiB\n",
                    (double)(i * object_size) /
                    (1024.0 * 1024.0 * 1024.0));

            release_host_objects(
                dram_object_pointers,
                allocated_objects);

            return EXIT_FAILURE;
        }

        allocated_objects++;

        fill_object(
            dram_object_pointers[i],
            object_size,
            i);

        if ((i + 1) % 10 == 0 ||
            i == 0 ||
            i + 1 == iterations) {
            printf("Prepared %zu/%zu objects, "
                   "%.3f GiB pinned DRAM\n",
                   i + 1,
                   iterations,
                   (double)((i + 1) * object_size) /
                   (1024.0 * 1024.0 * 1024.0));
            fflush(stdout);
        }
    }

    /*
     * 确认 cudaHostAlloc 返回的是 Host memory，
     * 而不是 GPU device memory。
     */
    cudaPointerAttributes pointer_attributes;

    CUDA_CHECK(cudaPointerGetAttributes(
        &pointer_attributes,
        dram_object_pointers[0]));

#if CUDART_VERSION >= 10000
    printf("\nFirst host object memory type: ");

    if (pointer_attributes.type ==
        cudaMemoryTypeHost) {
        printf("Host / pinned DRAM\n");
    } else if (pointer_attributes.type ==
               cudaMemoryTypeDevice) {
        printf("Device / GPU memory\n");
    } else if (pointer_attributes.type ==
               cudaMemoryTypeManaged) {
        printf("Managed memory\n");
    } else {
        printf("Other (%d)\n",
               (int)pointer_attributes.type);
    }
#endif

    /*
     * GPU 端：一次性分配所有 object 的空间。
     *
     * Layout:
     *
     * [object 0][object 1]...[object N-1]
     */
    unsigned char *gpu_buffer = NULL;

    size_t gpu_free_before = 0;
    size_t gpu_free_after = 0;
    size_t gpu_total = 0;

    CUDA_CHECK(cudaMemGetInfo(
        &gpu_free_before,
        &gpu_total));

    printf("Allocating complete GPU dataset buffer...\n");

    CUDA_CHECK(cudaMalloc(
        (void **)&gpu_buffer,
        total_buffer_size));

    CUDA_CHECK(cudaMemGetInfo(
        &gpu_free_after,
        &gpu_total));

    printf("GPU buffer address    : %p\n",
           (void *)gpu_buffer);
    printf("Requested allocation  : %.3f GiB\n",
           (double)total_buffer_size /
           (1024.0 * 1024.0 * 1024.0));
    printf("Observed allocation   : %.3f GiB\n",
           (double)(gpu_free_before -
                    gpu_free_after) /
           (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking));

    /*
     * Warm-up 使用 object 0，并写到 GPU object 0 的位置。
     * Warm-up 不计入正式统计。
     */
    if (warmup_iterations > 0) {
        printf("Running %zu warm-up copies...\n",
               warmup_iterations);

        for (size_t i = 0;
             i < warmup_iterations;
             i++) {
            size_t source_index =
                i % iterations;

            CUDA_CHECK(cudaMemcpyAsync(
                gpu_buffer,
                dram_object_pointers[source_index],
                object_size,
                cudaMemcpyHostToDevice,
                stream));

            CUDA_CHECK(cudaStreamSynchronize(
                stream));
        }
    }

    double *latencies_us =
        (double *)malloc(
            iterations * sizeof(double));

    double *sorted_latencies_us =
        (double *)malloc(
            iterations * sizeof(double));

    if (latencies_us == NULL ||
        sorted_latencies_us == NULL) {
        fprintf(stderr,
                "Failed to allocate latency arrays\n");

        free(latencies_us);
        free(sorted_latencies_us);

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(gpu_buffer));

        release_host_objects(
            dram_object_pointers,
            allocated_objects);

        return EXIT_FAILURE;
    }

    printf("Starting pinned DRAM -> GPU latency test...\n");

    uint64_t total_start_ns =
        get_time_ns();

    size_t maximum_latency_index = 0;
    double observed_maximum_us = 0.0;

    for (size_t i = 0; i < iterations; i++) {
        /*
         * 每个 object 写入 GPU buffer 中不同的位置。
         */
        unsigned char *gpu_destination =
            gpu_buffer + i * object_size;

        uint64_t operation_start_ns =
            get_time_ns();

        CUDA_CHECK(cudaMemcpyAsync(
            gpu_destination,
            dram_object_pointers[i],
            object_size,
            cudaMemcpyHostToDevice,
            stream));

        /*
         * 等待本次 object 完全传输结束。
         * 保持 outstanding operation = 1。
         */
        CUDA_CHECK(cudaStreamSynchronize(
            stream));

        uint64_t operation_end_ns =
            get_time_ns();

        latencies_us[i] =
            (double)(
                operation_end_ns -
                operation_start_ns
            ) / 1000.0;

        if (latencies_us[i] >
            observed_maximum_us) {
            observed_maximum_us =
                latencies_us[i];

            maximum_latency_index = i;
        }

        if ((i + 1) % 100 == 0 ||
            i + 1 == iterations) {
            printf("Completed %zu/%zu copies\n",
                   i + 1,
                   iterations);
            fflush(stdout);
        }
    }

    uint64_t total_end_ns =
        get_time_ns();

    /*
     * 统计 latency。
     */
    memcpy(sorted_latencies_us,
           latencies_us,
           iterations * sizeof(double));

    qsort(sorted_latencies_us,
          iterations,
          sizeof(double),
          compare_double);

    double minimum_us =
        sorted_latencies_us[0];

    double maximum_us =
        sorted_latencies_us[iterations - 1];

    double typical_us =
        calculate_median(
            sorted_latencies_us,
            iterations);

    double average_us =
        calculate_average(
            latencies_us,
            iterations);

    double stdev_us =
        calculate_sample_stdev(
            latencies_us,
            iterations,
            average_us);

    double p90_us =
        percentile_nearest_rank(
            sorted_latencies_us,
            iterations,
            90.0);

    double p95_us =
        percentile_nearest_rank(
            sorted_latencies_us,
            iterations,
            95.0);

    double p99_us =
        percentile_nearest_rank(
            sorted_latencies_us,
            iterations,
            99.0);

    double p999_us =
        percentile_nearest_rank(
            sorted_latencies_us,
            iterations,
            99.9);

    double total_elapsed_seconds =
        (double)(
            total_end_ns -
            total_start_ns
        ) / 1000000000.0;

    double effective_bandwidth_gb_s =
        (double)total_buffer_size /
        total_elapsed_seconds /
        1.0e9;

    double effective_bandwidth_gbit_s =
        effective_bandwidth_gb_s * 8.0;

    printf("\n");
    printf("====================================================================================================================================================\n");
    printf(" #bytes       #iterations   min[usec]   max[usec]   typical[usec]   avg[usec]   stdev[usec]   p99[usec]   p99.9[usec]\n");
    printf(" %-12zu %-13zu %-11.3f %-11.3f %-15.3f %-11.3f %-13.3f %-11.3f %.3f\n",
           object_size,
           iterations,
           minimum_us,
           maximum_us,
           typical_us,
           average_us,
           stdev_us,
           p99_us,
           p999_us);
    printf("====================================================================================================================================================\n");

    printf("P90 latency           : %.3f usec\n",
           p90_us);
    printf("P95 latency           : %.3f usec\n",
           p95_us);
    printf("P99 latency           : %.3f usec\n",
           p99_us);
    printf("P99.9 latency         : %.3f usec\n",
           p999_us);
    printf("Maximum latency index : %zu\n",
           maximum_latency_index);

    printf("\n");

    printf("Minimum latency       : %.6f ms\n",
           minimum_us / 1000.0);
    printf("Maximum latency       : %.6f ms\n",
           maximum_us / 1000.0);
    printf("Typical latency       : %.6f ms\n",
           typical_us / 1000.0);
    printf("Average latency       : %.6f ms\n",
           average_us / 1000.0);
    printf("Standard deviation    : %.6f ms\n",
           stdev_us / 1000.0);
    printf("P99 latency           : %.6f ms\n",
           p99_us / 1000.0);
    printf("P99.9 latency         : %.6f ms\n",
           p999_us / 1000.0);

    printf("\n");

    printf("Total elapsed time    : %.6f seconds\n",
           total_elapsed_seconds);
    printf("Effective bandwidth   : %.3f GB/s\n",
           effective_bandwidth_gb_s);
    printf("Effective bandwidth   : %.3f Gb/s\n",
           effective_bandwidth_gbit_s);

    /*
     * 验证每个 GPU object 开头的 index。
     */
    printf("\nVerifying GPU objects...\n");

    size_t verification_failures = 0;

    for (size_t i = 0; i < iterations; i++) {
        uint64_t copied_index =
            UINT64_MAX;

        CUDA_CHECK(cudaMemcpy(
            &copied_index,
            gpu_buffer + i * object_size,
            sizeof(copied_index),
            cudaMemcpyDeviceToHost));

        if (copied_index != (uint64_t)i) {
            fprintf(stderr,
                    "Verification failed at object %zu: "
                    "expected %zu, got %llu\n",
                    i,
                    i,
                    (unsigned long long)copied_index);

            verification_failures++;

            if (verification_failures >= 10) {
                fprintf(stderr,
                        "Too many verification failures; "
                        "stopping verification.\n");
                break;
            }
        }
    }

    if (verification_failures == 0) {
        printf("All %zu GPU objects verified successfully.\n",
               iterations);
    } else {
        printf("Verification failures: %zu\n",
               verification_failures);
    }

    printf("\n");
    printf("Memory source         : pinned host DRAM\n");
    printf("Destination           : GPU device memory\n");
    printf("GPU storage mode      : all objects retained\n");
    printf("GPU layout            : contiguous object array\n");
    printf("Copy direction        : cudaMemcpyHostToDevice\n");
    printf("Typical latency       : median / P50\n");
    printf("Outstanding copies    : 1\n");

    free(latencies_us);
    free(sorted_latencies_us);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(gpu_buffer));

    release_host_objects(
        dram_object_pointers,
        allocated_objects);

    return verification_failures == 0
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}