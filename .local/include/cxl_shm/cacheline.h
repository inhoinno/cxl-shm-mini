#ifndef __CXL_SHM_CACHELINE_H__
#define __CXL_SHM_CACHELINE_H__

#include <stdint.h>
#include <sys/types.h>

/* cacheline helper implementation */

#define CACHELINE_SIZE 64

#define PAGE_SHIFT 12
#define PAGE_SIZE (1 << PAGE_SHIFT)

#define __algn__(size)         __attribute__((aligned(size)))
#define __align_cacheline__     __algn__(CACHELINE_SIZE)
#define __align_page__          __algn__(PAGE_SIZE)

void clflush_region_with_mfence(void *addr, size_t size);
void clflush_region_with_sfence(void *addr, size_t size);

void clwb_region_with_barrier(void *addr, size_t size);

#endif /* __CXL_SHM_CACHELINE_H__ */