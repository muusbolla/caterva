/*
 * Copyright (C) 2018 Francesc Alted, Aleix Alcacer.
 * Copyright (C) 2019-present Blosc Development team <blosc@blosc.org>
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include "test_common.h"

typedef struct {
    int8_t ndim;
    int64_t shape[CATERVA_MAX_DIM];
    int32_t chunkshape[CATERVA_MAX_DIM];
    int32_t blockshape[CATERVA_MAX_DIM];
    int32_t chunkshape2[CATERVA_MAX_DIM];
    int32_t blockshape2[CATERVA_MAX_DIM];
    int64_t start[CATERVA_MAX_DIM];
    int64_t stop[CATERVA_MAX_DIM];
} test_squeeze_shapes_t;


CUTEST_TEST_DATA(squeeze) {
    caterva_ctx_t *ctx;
};


CUTEST_TEST_SETUP(squeeze) {
    caterva_config_t cfg = CATERVA_CONFIG_DEFAULTS;
    cfg.nthreads = 2;
    cfg.compcodec = BLOSC_BLOSCLZ;
    caterva_ctx_new(&cfg, &data->ctx);

    // Add parametrizations
    CUTEST_PARAMETRIZE(itemsize, uint8_t, CUTEST_DATA(
            1,
            2,
            4,
            8
    ));
    CUTEST_PARAMETRIZE(backend, _test_backend, CUTEST_DATA(
            {CATERVA_STORAGE_PLAINBUFFER, false, false},
            {CATERVA_STORAGE_BLOSC, false, false},
            {CATERVA_STORAGE_BLOSC, true, false},
            {CATERVA_STORAGE_BLOSC, true, true},
    ));
    CUTEST_PARAMETRIZE(backend2, _test_backend, CUTEST_DATA(
            {CATERVA_STORAGE_PLAINBUFFER, false, false},
            {CATERVA_STORAGE_BLOSC, false, false},
            {CATERVA_STORAGE_BLOSC, true, false},
            {CATERVA_STORAGE_BLOSC, true, true},
    ));


    CUTEST_PARAMETRIZE(shapes, test_squeeze_shapes_t, CUTEST_DATA(
            {0, {0}, {0}, {0}, {0}, {0}, {0}, {0}}, // 0-dim
            {1, {10}, {7}, {2}, {1}, {1}, {2}, {3}}, // 1-idim
            {2, {14, 10}, {8, 5}, {2, 2}, {4, 1}, {2, 1}, {5, 3}, {9, 4}}, // general,
            {3, {10, 10, 10}, {3, 5, 9}, {3, 4, 4}, {1, 7, 1}, {1, 5, 1}, {3, 0, 9}, {4, 7, 10}},
            {2, {20, 0}, {7, 0}, {3, 0}, {1, 0}, {1, 0}, {1, 0}, {2, 0}}, // 0-shape
            {2, {20, 10}, {7, 5}, {3, 5}, {1, 0}, {1, 0}, {17, 0}, {18, 0}}, // 0-shape
    ));
}


CUTEST_TEST_TEST(squeeze) {
    CUTEST_GET_PARAMETER(backend, _test_backend);
    CUTEST_GET_PARAMETER(shapes, test_squeeze_shapes_t);
    CUTEST_GET_PARAMETER(backend2, _test_backend);
    CUTEST_GET_PARAMETER(itemsize, uint8_t);
    caterva_params_t params;
    params.itemsize = itemsize;
    params.ndim = shapes.ndim;
    for (int i = 0; i < params.ndim; ++i) {
        params.shape[i] = shapes.shape[i];
    }

    caterva_storage_t storage = {0};
    storage.backend = backend.backend;
    switch (storage.backend) {
        case CATERVA_STORAGE_PLAINBUFFER:
            break;
        case CATERVA_STORAGE_BLOSC:
            if (backend.persistent) {
                storage.properties.blosc.urlpath = "test_squeeze.b2frame";
            }
            storage.properties.blosc.sequencial = backend.sequential;
            for (int i = 0; i < params.ndim; ++i) {
                storage.properties.blosc.chunkshape[i] = shapes.chunkshape[i];
                storage.properties.blosc.blockshape[i] = shapes.blockshape[i];
            }
            break;
        default:
            CATERVA_TEST_ASSERT(CATERVA_ERR_INVALID_STORAGE);
    }

    /* Create original data */
    size_t buffersize = itemsize;
    for (int i = 0; i < params.ndim; ++i) {
        buffersize *= (size_t) params.shape[i];
    }
    uint8_t *buffer = malloc(buffersize);
    CUTEST_ASSERT("Buffer filled incorrectly", fill_buf(buffer, itemsize, buffersize / itemsize));

    /* Create caterva_array_t with original data */
    caterva_array_t *src;
    CATERVA_TEST_ASSERT(caterva_from_buffer(data->ctx, buffer, buffersize, &params, &storage,
                                            &src));


    /* Create storage for dest container */

    caterva_storage_t storage2 = {0};
    storage2.backend = backend2.backend;
    switch (storage2.backend) {
        case CATERVA_STORAGE_PLAINBUFFER:
            break;
        case CATERVA_STORAGE_BLOSC:
            if (backend2.persistent) {
                storage2.properties.blosc.urlpath = "test_sequeeze2.b2frame";
            }
            storage2.properties.blosc.sequencial = backend2.sequential;
            for (int i = 0; i < params.ndim; ++i) {
                storage2.properties.blosc.chunkshape[i] = shapes.chunkshape2[i];
                storage2.properties.blosc.blockshape[i] = shapes.blockshape2[i];
            }
            break;
        default:
            CATERVA_TEST_ASSERT(CATERVA_ERR_INVALID_STORAGE);
    }

    caterva_array_t *dest;
    CATERVA_TEST_ASSERT(caterva_get_slice(data->ctx, src, shapes.start, shapes.stop,
                                          &storage2, &dest));

    CATERVA_TEST_ASSERT(caterva_squeeze(data->ctx, dest));

    if (params.ndim != 0) {
        CUTEST_ASSERT("dims are equal", src->ndim != dest->ndim);
    }

    free(buffer);
    CATERVA_TEST_ASSERT(caterva_free(data->ctx, &src));
    CATERVA_TEST_ASSERT(caterva_free(data->ctx, &dest));
    
    return 0;
}

CUTEST_TEST_TEARDOWN(squeeze) {
    caterva_ctx_free(&data->ctx);
}

int main() {
    CUTEST_TEST_RUN(squeeze);
}
