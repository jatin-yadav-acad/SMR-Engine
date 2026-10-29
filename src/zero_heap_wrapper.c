#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void* __real_malloc(size_t size);
extern void __real_free(void* ptr);

static size_t dynamic_allocations_count = 0;
static size_t dynamic_bytes_requested = 0;

void* __wrap_malloc(size_t size) {
    dynamic_allocations_count++;
    dynamic_bytes_requested += size;
    fprintf(stderr, "[HARDWARE VIOLATION] Illegal dynamic malloc: %zu bytes requested!\n", size);
#ifdef ENFORCE_MISRA_STRICT
    abort();
#endif
    return __real_malloc(size);
}

void __wrap_free(void* ptr) {
    __real_free(ptr);
}

void print_heap_audit(void) {
    fprintf(stdout, "[AUDIT] Total dynamic allocations intercepted: %zu (%zu bytes)\n",
            dynamic_allocations_count, dynamic_bytes_requested);
}

size_t get_dynamic_allocations_count(void) {
    return dynamic_allocations_count;
}

size_t get_dynamic_bytes_requested(void) {
    return dynamic_bytes_requested;
}

void reset_heap_audit(void) {
    dynamic_allocations_count = 0;
    dynamic_bytes_requested = 0;
}

#ifdef __cplusplus
}
#endif
