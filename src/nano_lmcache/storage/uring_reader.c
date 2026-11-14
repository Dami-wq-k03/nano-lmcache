/* Linux syscall-only io_uring reads; one private ring and fixed buffer per call. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#ifdef __linux__
#include <linux/io_uring.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#define LOAD(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define STORE(p, v) __atomic_store_n((p), (v), __ATOMIC_RELEASE)

int nlc_uring_probe(void) {
    struct io_uring_params p = {0};
    int ring = syscall(SYS_io_uring_setup, 1, &p);
    if (ring < 0) return -errno;
    close(ring);
    return 0;
}

int nlc_uring_read(int fd, unsigned count, const uint64_t *offsets,
                   const uint32_t *lengths, unsigned depth, void *output,
                   int32_t *results) {
    struct io_uring_params p = {0};
    void *sq = MAP_FAILED, *cq = MAP_FAILED, *data = MAP_FAILED;
    struct io_uring_sqe *sqes = MAP_FAILED;
    size_t sqsize = 0, cqsize = 0, total = 0;
    int ring = -1, error = 0;
    if (!count || count > 1024 || !depth || depth > 256) return -EINVAL;
    for (unsigned i = 0; i < count; i++) {
        if (lengths[i] > INT32_MAX || offsets[i] > INT64_MAX) return -EINVAL;
        total += lengths[i];
        results[i] = INT32_MIN;
    }
    if (total > (64u << 20)) return -E2BIG;
    ring = syscall(SYS_io_uring_setup, depth, &p);
    if (ring < 0) return -errno;
    sqsize = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    cqsize = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        if (cqsize > sqsize) sqsize = cqsize;
        cq = sq = mmap(NULL, sqsize, PROT_READ | PROT_WRITE, MAP_SHARED,
                       ring, IORING_OFF_SQ_RING);
    } else {
        sq = mmap(NULL, sqsize, PROT_READ | PROT_WRITE, MAP_SHARED,
                  ring, IORING_OFF_SQ_RING);
        cq = mmap(NULL, cqsize, PROT_READ | PROT_WRITE, MAP_SHARED,
                  ring, IORING_OFF_CQ_RING);
    }
    sqes = mmap(NULL, p.sq_entries * sizeof(*sqes), PROT_READ | PROT_WRITE,
                MAP_SHARED, ring, IORING_OFF_SQES);
    data = mmap(NULL, total ? total : 1, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED ||
        data == MAP_FAILED) { error = -errno; goto cleanup; }
    struct iovec buffer = {.iov_base = data, .iov_len = total ? total : 1};
    if (syscall(SYS_io_uring_register, ring, IORING_REGISTER_BUFFERS,
                &buffer, 1) < 0) { error = -errno; goto cleanup; }
    unsigned *sqhead = (void *)((char *)sq + p.sq_off.head);
    unsigned *sqtail = (void *)((char *)sq + p.sq_off.tail);
    unsigned *sqmask = (void *)((char *)sq + p.sq_off.ring_mask);
    unsigned *array = (void *)((char *)sq + p.sq_off.array);
    unsigned *cqhead = (void *)((char *)cq + p.cq_off.head);
    unsigned *cqtail = (void *)((char *)cq + p.cq_off.tail);
    unsigned *cqmask = (void *)((char *)cq + p.cq_off.ring_mask);
    struct io_uring_cqe *cqes = (void *)((char *)cq + p.cq_off.cqes);
    size_t position = 0;
    for (unsigned start = 0; start < count;) {
        unsigned batch = count - start < depth ? count - start : depth;
        unsigned tail = LOAD(sqtail);
        if (tail - LOAD(sqhead) + batch > p.sq_entries) {
            error = -EIO; goto cleanup;
        }
        for (unsigned i = 0; i < batch; i++) {
            unsigned slot = (tail + i) & *sqmask;
            struct io_uring_sqe *entry = &sqes[slot];
            memset(entry, 0, sizeof(*entry));
            entry->opcode = IORING_OP_READ_FIXED;
            entry->fd = fd;
            entry->off = offsets[start + i];
            entry->addr = (uintptr_t)data + position;
            entry->len = lengths[start + i];
            entry->user_data = start + i;
            array[slot] = slot;
            position += entry->len;
        }
        STORE(sqtail, tail + batch);
        for (unsigned pending = batch; pending;) {
            int n = syscall(SYS_io_uring_enter, ring, pending, 0, 0, NULL, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { error = n < 0 ? -errno : -EIO; goto cleanup; }
            pending -= (unsigned)n;
        }
        for (unsigned done = 0; done < batch;) {
            unsigned head = LOAD(cqhead);
            if (head == LOAD(cqtail)) {
                int n = syscall(SYS_io_uring_enter, ring, 0, 1,
                                IORING_ENTER_GETEVENTS, NULL, 0);
                if (n < 0 && errno != EINTR) { error = -errno; goto cleanup; }
                continue;
            }
            struct io_uring_cqe *entry = &cqes[head & *cqmask];
            uint64_t index = entry->user_data;
            if (index < start || index >= start + batch ||
                results[index] != INT32_MIN) { error = -EIO; goto cleanup; }
            results[index] = entry->res;
            STORE(cqhead, head + 1);
            done++;
        }
        start += batch;
    }
    memcpy(output, data, total);
cleanup:
    /* Fixed I/O holds pinned page refs. On failure, unmapping cannot recycle
       these private pages while ring teardown is still cancelling pending I/O. */
    close(ring);
    if (sqes != MAP_FAILED) munmap(sqes, p.sq_entries * sizeof(*sqes));
    if (cq != MAP_FAILED && cq != sq) munmap(cq, cqsize);
    if (sq != MAP_FAILED) munmap(sq, sqsize);
    if (data != MAP_FAILED) munmap(data, total ? total : 1);
    return error;
}
#else
/* Allows local compiler sanity checks, but never claims non-Linux support. */
int nlc_uring_probe(void) { return -ENOSYS; }
int nlc_uring_read(int fd, unsigned count, const uint64_t *offsets,
                   const uint32_t *lengths, unsigned depth, void *output,
                   int32_t *results) {
    (void)fd; (void)count; (void)offsets; (void)lengths;
    (void)depth; (void)output; (void)results;
    return -ENOSYS;
}
#endif
