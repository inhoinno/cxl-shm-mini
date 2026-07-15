// dax_memcpy_bw.c

#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define MB (1024ULL * 1024ULL)

static double now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    const char *dev = (argc > 1) ? argv[1] : "/dev/dax1.0";

    const uint64_t start_offset = 64ULL * GB;
    const uint64_t read_size    = 32ULL * GB;
    const uint64_t chunk_size   = 1ULL * MB;

    const uint64_t map_size = start_offset + read_size;

    int fd = open(dev, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    void *base = mmap(NULL, map_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    char *src = (char *)base + start_offset;

    void *buf = NULL;
    if (posix_memalign(&buf, 4096, chunk_size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        munmap(base, map_size);
        close(fd);
        return 1;
    }

    // Pre-touch buffer so page allocation/page fault is not counted.
    memset(buf, 0, chunk_size);

    printf("Device          : %s\n", dev);
    printf("Start Offset    : %.2f GB\n", start_offset / (double)GB);
    printf("Read Size       : %.2f GB\n", read_size / (double)GB);
    printf("Chunk Size      : %.2f MB\n", chunk_size / (double)MB);

    volatile uint64_t checksum = 0;

    double t0 = now_sec();

    for (uint64_t off = 0; off < read_size; off += chunk_size) {
        memcpy(buf, src + off, chunk_size);

        // Prevent compiler from fully optimizing away the memcpy result.
        checksum += *(volatile uint64_t *)buf;
    }

    double t1 = now_sec();

    double elapsed = t1 - t0;
    double bw = (read_size / (double)GB) / elapsed;

    printf("Completion Time : %.6f sec\n", elapsed);
    printf("Bandwidth       : %.2f GB/s\n", bw);
    printf("Checksum        : %" PRIu64 "\n", checksum);

    free(buf);
    munmap(base, map_size);
    close(fd);

    return 0;
}