/*
 * Copyright (C) 2018-present Francesc Alted, Aleix Alcacer.
 * Copyright (C) 2019-present Blosc Development team <blosc@blosc.org>
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */
#include <caterva_utils.h>

void index_unidim_to_multidim(int8_t ndim, int64_t *shape, int64_t i, int64_t *index) {
    int64_t strides[CATERVA_MAX_DIM];
    if (ndim == 0) {
        return;
    }
    strides[ndim - 1] = 1;
    for (int j = ndim - 2; j >= 0; --j) {
        strides[j] = shape[j + 1] * strides[j + 1];
    }

    index[0] = i / strides[0];
    for (int j = 1; j < ndim; ++j) {
        index[j] = (i % strides[j - 1]) / strides[j];
    }
}

void index_multidim_to_unidim(int64_t *index, int8_t ndim, int64_t *strides, int64_t *i) {
    *i = 0;
    for (int j = 0; j < ndim; ++j) {
        *i += index[j] * strides[j];
    }
}

// big <-> little-endian and store it in a memory position.  Sizes supported: 1, 2, 4, 8 bytes.
void swap_store(void *dest, const void *pa, int size) {
    uint8_t *pa_ = (uint8_t *) pa;
    uint8_t *pa2_ = (uint8_t *)malloc((size_t) size);
    int i = 1; /* for big/little endian detection */
    char *p = (char *) &i;

    if (p[0] == 1) {
        /* little endian */
        switch (size) {
            case 8:
                pa2_[0] = pa_[7];
                pa2_[1] = pa_[6];
                pa2_[2] = pa_[5];
                pa2_[3] = pa_[4];
                pa2_[4] = pa_[3];
                pa2_[5] = pa_[2];
                pa2_[6] = pa_[1];
                pa2_[7] = pa_[0];
                break;
            case 4:
                pa2_[0] = pa_[3];
                pa2_[1] = pa_[2];
                pa2_[2] = pa_[1];
                pa2_[3] = pa_[0];
                break;
            case 2:
                pa2_[0] = pa_[1];
                pa2_[1] = pa_[0];
                break;
            case 1:
                pa2_[0] = pa_[0];
                break;
            default:
                fprintf(stderr, "Unhandled nitems: %d\n", size);
        }
    }
    memcpy(dest, pa2_, size);
    free(pa2_);
}

int32_t serialize_meta(uint8_t ndim, int64_t *shape, const int32_t *chunkshape,
                              const int32_t *blockshape, uint8_t **smeta) {
    // Allocate space for Caterva metalayer
    int32_t max_smeta_len = 1 + 1 + 1 + (1 + ndim * (1 + sizeof(int64_t))) +
                            (1 + ndim * (1 + sizeof(int32_t))) + (1 + ndim * (1 + sizeof(int32_t)));
    *smeta = (uint8_t*)malloc((size_t) max_smeta_len);
    CATERVA_ERROR_NULL(smeta);
    uint8_t *pmeta = *smeta;

    // Build an array with 5 entries (version, ndim, shape, chunkshape, blockshape)
    *pmeta++ = 0x90 + 5;

    // version entry
    *pmeta++ = CATERVA_METALAYER_VERSION;  // positive fixnum (7-bit positive integer)

    // ndim entry
    *pmeta++ = (uint8_t) ndim;  // positive fixnum (7-bit positive integer)

    // shape entry
    *pmeta++ = (uint8_t)(0x90) + ndim;  // fix array with ndim elements
    for (uint8_t i = 0; i < ndim; i++) {
        *pmeta++ = 0xd3;  // int64
        swap_store(pmeta, shape + i, sizeof(int64_t));
        pmeta += sizeof(int64_t);
    }

    // chunkshape entry
    *pmeta++ = (uint8_t)(0x90) + ndim;  // fix array with ndim elements
    for (uint8_t i = 0; i < ndim; i++) {
        *pmeta++ = 0xd2;  // int32
        swap_store(pmeta, chunkshape + i, sizeof(int32_t));
        pmeta += sizeof(int32_t);
    }

    // blockshape entry
    *pmeta++ = (uint8_t)(0x90) + ndim;  // fix array with ndim elements
    for (uint8_t i = 0; i < ndim; i++) {
        *pmeta++ = 0xd2;  // int32
        swap_store(pmeta, blockshape + i, sizeof(int32_t));
        pmeta += sizeof(int32_t);
    }
    int32_t slen = (int32_t)(pmeta - *smeta);

    return slen;
}

int32_t deserialize_meta(uint8_t *smeta, uint32_t smeta_len, uint8_t *ndim, int64_t *shape,
                                int32_t *chunkshape, int32_t *blockshape) {
    uint8_t *pmeta = smeta;
    CATERVA_UNUSED_PARAM(smeta_len);

    // Check that we have an array with 5 entries (version, ndim, shape, chunkshape, blockshape)
    pmeta += 1;

    // version entry
    int8_t version = pmeta[0];  // positive fixnum (7-bit positive integer)
    CATERVA_UNUSED_PARAM(version);

    pmeta += 1;

    // ndim entry
    *ndim = pmeta[0];
    int8_t ndim_aux = *ndim;  // positive fixnum (7-bit positive integer)
    pmeta += 1;

    // shape entry
    // Initialize to ones, as required by Caterva
    for (int i = 0; i < CATERVA_MAX_DIM; i++) shape[i] = 1;
    pmeta += 1;
    for (int8_t i = 0; i < ndim_aux; i++) {
        pmeta += 1;
        swap_store(shape + i, pmeta, sizeof(int64_t));
        pmeta += sizeof(int64_t);
    }

    // chunkshape entry
    // Initialize to ones, as required by Caterva
    for (int i = 0; i < CATERVA_MAX_DIM; i++) chunkshape[i] = 1;
    pmeta += 1;
    for (int8_t i = 0; i < ndim_aux; i++) {
        pmeta += 1;
        swap_store(chunkshape + i, pmeta, sizeof(int32_t));
        pmeta += sizeof(int32_t);
    }

    // blockshape entry
    // Initialize to ones, as required by Caterva
    for (int i = 0; i < CATERVA_MAX_DIM; i++) blockshape[i] = 1;
    pmeta += 1;
    for (int8_t i = 0; i < ndim_aux; i++) {
        pmeta += 1;
        swap_store(blockshape + i, pmeta, sizeof(int32_t));
        pmeta += sizeof(int32_t);
    }
    uint32_t slen = (uint32_t)(pmeta - smeta);
    CATERVA_UNUSED_PARAM(slen);

    return 0;
}

int caterva_copy_buffer(uint8_t ndim,
                        uint8_t itemsize,
                        void *src, int64_t *src_pad_shape,
                        int64_t *src_start, int64_t *src_stop,
                        void *dst, int64_t *dst_pad_shape,
                        int64_t *dst_start) {
switch(ndim) {
    case 1:
        caterva_copy_buffer<1>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    case 2:
        caterva_copy_buffer<2>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    case 3:
        caterva_copy_buffer<3>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    case 4:
        caterva_copy_buffer<4>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    case 5:
        caterva_copy_buffer<5>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    case 6:
        caterva_copy_buffer<6>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    case 7:
        caterva_copy_buffer<7>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    case 8:
        caterva_copy_buffer<8>(itemsize, src, src_pad_shape, src_start, src_stop, dst, dst_pad_shape, dst_start);
    break;
    default:
        // guard against potential future increase to CATERVA_MAX_DIM
        return CATERVA_ERR_INVALID_INDEX;
    break;
    }
    
    return CATERVA_SUCCEED;
}


int create_blosc_params(caterva_ctx_t *ctx,
                        caterva_params_t *params,
                        caterva_storage_t *storage,
                        blosc2_cparams *cparams,
                        blosc2_dparams *dparams,
                        blosc2_storage *b_storage) {
    int32_t blocknitems = 1;
    for (int i = 0; i < params->ndim; ++i) {
        blocknitems *= storage->blockshape[i];
    }

    memcpy(cparams, &BLOSC2_CPARAMS_DEFAULTS, sizeof(blosc2_cparams));
    cparams->blocksize = blocknitems * params->itemsize;
    cparams->schunk = NULL;
    cparams->typesize = params->itemsize;
    cparams->prefilter = ctx->cfg->prefilter;
    cparams->preparams = ctx->cfg->pparams;
    cparams->use_dict = ctx->cfg->usedict;
    cparams->nthreads = (int16_t) ctx->cfg->nthreads;
    cparams->clevel = (uint8_t) ctx->cfg->complevel;
    cparams->compcode = (uint8_t) ctx->cfg->compcodec;
    cparams->compcode_meta = (uint8_t) ctx->cfg->compmeta;
    for (int i = 0; i < BLOSC2_MAX_FILTERS; ++i) {
        cparams->filters[i] = ctx->cfg->filters[i];
        cparams->filters_meta[i] = ctx->cfg->filtersmeta[i];
    }
    cparams->udbtune = ctx->cfg->udbtune;
    cparams->splitmode = ctx->cfg->splitmode;

    memcpy(dparams, &BLOSC2_DPARAMS_DEFAULTS, sizeof(blosc2_dparams));
    dparams->schunk = NULL;
    dparams->nthreads = ctx->cfg->nthreads;

    memcpy(b_storage, &BLOSC2_STORAGE_DEFAULTS, sizeof(blosc2_storage));
    b_storage->cparams = cparams;
    b_storage->dparams = dparams;

    if (storage->sequencial) {
        b_storage->contiguous = true;
    }
    if (storage->urlpath != NULL) {
        b_storage->urlpath = storage->urlpath;
    }

    return CATERVA_SUCCEED;
}


int caterva_config_from_schunk(caterva_ctx_t *ctx, blosc2_schunk *sc, caterva_config_t *cfg) {
    cfg->alloc = ctx->cfg->alloc;
    cfg->free = ctx->cfg->free;

    cfg->complevel = sc->storage->cparams->clevel;
    cfg->compcodec = sc->storage->cparams->compcode;
    cfg->compmeta = sc->storage->cparams->compcode_meta;
    cfg->usedict = sc->storage->cparams->use_dict;
    cfg->splitmode = sc->storage->cparams->splitmode;
    cfg->nthreads = ctx->cfg->nthreads;
    for (int i = 0; i < BLOSC2_MAX_FILTERS; ++i) {
        cfg->filters[i] = sc->storage->cparams->filters[i];
        cfg->filtersmeta[i] = sc->storage->cparams->filters_meta[i];
    }

    cfg->prefilter = ctx->cfg->prefilter;
    cfg->pparams = ctx->cfg->pparams;
    cfg->udbtune = ctx->cfg->udbtune;

    return CATERVA_SUCCEED;
}
