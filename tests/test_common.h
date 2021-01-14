//
// Created by Aleix Alcacer Sales on 07/06/2020.
//

#ifndef CATERVA_TEST_COMMON_H
#define CATERVA_TEST_COMMON_H

#include <caterva.h>
#include "cutest.h"


#define CATERVA_TEST_ASSERT(rc) CUTEST_ASSERT(print_error(rc), (rc) == CATERVA_SUCCEED);

#ifdef __GNUC__
#define CATERVA_TEST_UNUSED __attribute__((unused))
#else
#define CATERVA_TEST_UNUSED
#endif
static bool fill_buf(void *buf, uint8_t itemsize, size_t buf_size) CATERVA_TEST_UNUSED;
        static bool fill_buf(void *buf, uint8_t itemsize, size_t buf_size) {
    switch (itemsize) {
        case 8:
            for (size_t i = 0; i < buf_size; ++i) {
                ((double *) buf)[i] = (double) i;
            }
            break;
        case 4:
            for (size_t i = 0; i < buf_size; ++i) {
                ((float *) buf)[i] = (float) i;
            }
            break;
        case 2:
            for (size_t i = 0; i < buf_size; ++i) {
                ((uint16_t *) buf)[i] = (uint16_t ) i;
            }
            break;
        case 1:
            for (size_t i = 0; i < buf_size; ++i) {
                ((uint8_t *) buf)[i] = (uint8_t ) i;
            }
            break;
        default:
            return false;
    }
    return true;
}


/* Tests data */

typedef struct {
    int8_t ndim;
    int64_t shape[CATERVA_MAX_DIM];
    int32_t chunkshape[CATERVA_MAX_DIM];
    int32_t blockshape[CATERVA_MAX_DIM];
} _test_shapes;


typedef struct {
    caterva_storage_backend_t backend;
    bool sequential;
    bool persistent;
} _test_backend;


void caterva_default_parameters() {
    CUTEST_PARAMETRIZE(itemsize, bool, CUTEST_DATA(1, 2, 4, 8));
    CUTEST_PARAMETRIZE(shapes, _test_shapes, CUTEST_DATA(
        {2, {100, 100}, {20, 20}, {10, 10}},
        {3, {100, 55, 123}, {31, 5, 22}, {4, 4, 4}},
        {3, {100, 0, 12}, {31, 0, 12}, {10, 0, 12}},
        {4, {50, 160, 31, 12}, {25, 20, 20, 10}, {5, 5, 5, 10}},
        {5, {1, 1, 1024, 1, 1}, {1, 1, 500, 1, 1}, {1, 1, 200, 1, 1}},
        {6, {5, 1, 200, 3, 1, 2}, {5, 1, 50, 2, 1, 2}, {2, 1, 20, 2, 1, 2}},
    ));
    CUTEST_PARAMETRIZE(backend, _test_backend, CUTEST_DATA(
        {CATERVA_STORAGE_PLAINBUFFER, false, false},
        {CATERVA_STORAGE_BLOSC, false, false},
        {CATERVA_STORAGE_BLOSC, true, false},
        {CATERVA_STORAGE_BLOSC, true, true},
    ));
}


#define CATERVA_TEST_ASSERT_BUFFER(buffer1, buffer2, buffersize) \
    for (int i = 0; i < (buffersize); ++i) { \
        CUTEST_ASSERT("elements are not equals!", (buffer1)[i] == (buffer2)[i]); \
    }


#endif //CATERVA_TEST_COMMON_H
