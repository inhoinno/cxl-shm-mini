//gcc cxl_write_20.c     -I/home/duo/.local/include     -L/hom
//e/duo/.local/lib/x86_64-linux-gnu     -lcxl_shm     -Wl,-rpath,/home/duo/.local/lib/x86_64-linux-gnu  
//   -o write_20     -g -O2

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <cxl_shm/cacheline.h>
#include <cxl_shm/api.h>

#define NUM_OBJECTS 1000

/*
 * 20.48 MB, decimal definition:
 * 20.48 * 1,000,000 = 20,480,000 bytes
 */
#define OBJECT_SIZE ((size_t)20480000)

static void make_key(char *key, size_t key_size, size_t index)
{
    snprintf(key, key_size, "object_%04zu", index);
}

/*
 * 给不同 object 填入不同的数据 pattern。
 * 后续可以在 GPU 拷回少量数据进行验证。
 */
static void fill_object(void *buffer, size_t size, size_t index)
{
    unsigned char pattern = (unsigned char)(index % 251);
    memset(buffer, pattern, size);

    /*
     * 把 object index 写入开头，便于进一步验证。
     */
    if (size >= sizeof(uint64_t)) {
        uint64_t value = (uint64_t)index;
        memcpy(buffer, &value, sizeof(value));
    }
}

int main(void)
{
    printf("Object count : %d\n", NUM_OBJECTS);
    printf("Object size  : %zu bytes\n", OBJECT_SIZE);
    printf("Total size   : %.3f GiB\n",
           ((double)NUM_OBJECTS * OBJECT_SIZE) /
           (1024.0 * 1024.0 * 1024.0));

    if (cxl_shm_init() != 0) {
        fprintf(stderr, "cxl_shm_init failed\n");
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < NUM_OBJECTS; i++) {
        char key[64];

        make_key(key, sizeof(key), i);

        void *buffer = shmalloc(OBJECT_SIZE);
        if (buffer == NULL) {
            fprintf(stderr,
                    "shmalloc failed at object %zu, allocated %.3f GiB\n",
                    i,
                    ((double)i * OBJECT_SIZE) /
                    (1024.0 * 1024.0 * 1024.0));
            return EXIT_FAILURE;
        }

        fill_object(buffer, OBJECT_SIZE, i);

        /*
         * Flush 整个 object，而不只是 strlen()。
         * 对二进制 object 必须使用完整 OBJECT_SIZE。
         */
        clflush_region_with_mfence(buffer, OBJECT_SIZE);

        if (cxl_shm_put(key, buffer) != 0) {
            fprintf(stderr, "cxl_shm_put failed for key %s\n", key);
            return EXIT_FAILURE;
        }

        if ((i + 1) % 10 == 0 || i == 0) {
            printf("Written %zu/%d objects, %.3f GiB\n",
                   i + 1,
                   NUM_OBJECTS,
                   ((double)(i + 1) * OBJECT_SIZE) /
                   (1024.0 * 1024.0 * 1024.0));
            fflush(stdout);
        }

        /*
         * 不可以在这里 shfree(buffer)，因为 cxl_shm_put()
         * 保存的很可能只是这个 object 的 offset/pointer。
         */
    }

    printf("All objects written successfully.\n");
    printf("Writer remains alive. Press Ctrl+C to terminate.\n");
    fflush(stdout);

    /*
     * 如果共享内存 allocator/metadata 在 writer 退出后仍然有效，
     * 可以删掉这个 pause()。
     */
    for (;;) {
        pause();
    }

    return EXIT_SUCCESS;
}