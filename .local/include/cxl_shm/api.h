/*
 * Copyright (C) by Sherry Wang
 */
#ifndef __CXL_SHM_API_H__
#define __CXL_SHM_API_H__

#include <stdint.h>
#include <sys/types.h>

typedef uint64_t shm_ptr_t;

/* initialization */
int cxl_shm_init(); 
int cxl_shm_connect(); 
int cxl_shm_finalize();
int cxl_shm_is_initialized();

/* Key_reference obj store API */
typedef const char * cxl_key_t;

/**
 * @brief Update an object
 * @param key
 * @param addr     local pointer to the object.
 *                 It must be allocated from the `shmalloc` function
 * @return Zero: success, Nonzero: failure
 */
int cxl_shm_put(cxl_key_t key, void* addr);

/**
 * @brief Find an object by key
 * @param key
 * @param addr [out]   local pointer to the object
 * @return Zero: success, Nonzero: failure
*/
int cxl_shm_get(cxl_key_t key, void** addr);

/**
 * @brief Destroy key. Before calling this function,
 *        do not free memory (`shfree`) but free all locks 
 *        (`cxl_shm_free_lock`) if you allocated lock
 *        in the object.
 * @param key
 * @return Zero: success, Nonzero: failure
 */
int cxl_shm_destroy(cxl_key_t key);

/* Raw memory allocation */
void *shmalloc(size_t size);

// /* page-aligned memory allocation */
// void *shmalloc_align(size_t size);

/* free memory */
void shfree(void *ptr);

/* Lock API */
typedef struct {
    shm_ptr_t lockptr;
} cxl_lock_t;

int cxl_shm_allocate_lock(cxl_lock_t *lock);
void cxl_shm_free_lock(cxl_lock_t lock);

int cxl_shm_lock_acquire(cxl_lock_t lock);
void cxl_shm_lock_release(cxl_lock_t lock);

/* offset <-> ptr translation */
shm_ptr_t cxl_shm_get_offset(void *ptr);
void *cxl_shm_get_ptr(shm_ptr_t off);

#endif /* __CXL_SHM_API_H__ */
