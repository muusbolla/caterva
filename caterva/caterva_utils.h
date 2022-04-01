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

#ifndef CATERVA_CATERVA_UTILS_H_
#define CATERVA_CATERVA_UTILS_H_

#include <caterva.h>

void index_unidim_to_multidim(int8_t ndim, int64_t *shape, int64_t i, int64_t *index);
void index_multidim_to_unidim(int64_t *index, int8_t ndim, int64_t *strides, int64_t *i);

int32_t serialize_meta(uint8_t ndim, int64_t *shape, const int32_t *chunkshape,
                       const int32_t *blockshape, uint8_t **smeta);
int32_t deserialize_meta(uint8_t *smeta, uint32_t smeta_len, uint8_t *ndim, int64_t *shape,
                         int32_t *chunkshape, int32_t *blockshape);

void swap_store(void *dest, const void *pa, int size);

int caterva_copy_buffer(uint8_t ndim,
                        uint8_t itemsize,
                        void *src, int64_t *src_pad_shape,
                        int64_t *src_start, int64_t *src_stop,
                        void *dst, int64_t *dst_pad_shape,
                        int64_t *dst_start);

int create_blosc_params(caterva_ctx_t *ctx,
                        caterva_params_t *params,
                        caterva_storage_t *storage,
                        blosc2_cparams *cparams,
                        blosc2_dparams *dparams,
                        blosc2_storage *b_storage);

int caterva_config_from_schunk(caterva_ctx_t *ctx, blosc2_schunk *sc, caterva_config_t *cfg);


// called by ndim_copy (don't call directly)
template<int ndim, int mydim>
void ndim_copy_loop(const uint8_t itemsize,
            const int64_t* copy_shape,
            const uint8_t *bsrc, const int64_t* src_strides, int64_t& src_copy_start,
            uint8_t *bdst, const int64_t* dst_strides, int64_t& dst_copy_start) {
  if constexpr(mydim == ndim) {
    int64_t copy_nbytes = copy_shape[ndim-1] * itemsize;
    memcpy(&bdst[dst_copy_start * itemsize], &bsrc[src_copy_start * itemsize], copy_nbytes);
  } else {
    int64_t copy_start = 0;
    do {
      ndim_copy_loop<ndim,mydim+1>(itemsize,copy_shape,bsrc,src_strides,src_copy_start,bdst,dst_strides,dst_copy_start);
      ++copy_start;
      if constexpr(mydim == ndim - 1) {
        src_copy_start += src_strides[mydim-1];
        dst_copy_start += dst_strides[mydim-1];
      } else {
        src_copy_start += src_strides[mydim-1] - copy_shape[mydim] * src_strides[mydim];
        dst_copy_start += dst_strides[mydim-1] - copy_shape[mydim] * dst_strides[mydim];

      }
    } while(copy_start < copy_shape[mydim-1]);
  }
}

// called by caterva_copy_buffer<ndim> (don't call directly)
template<int ndim>
void ndim_copy(const uint8_t itemsize,
            const int64_t* copy_shape,
            const uint8_t *bsrc, const int64_t* src_strides,
            uint8_t *bdst, const int64_t* dst_strides) {
  int64_t src_copy_start = 0;
  int64_t dst_copy_start = 0;
  ndim_copy_loop<ndim, 1>(itemsize, copy_shape, bsrc, src_strides, src_copy_start, bdst, dst_strides, dst_copy_start);
}

// copy n-dimensional data, translating between source and destination strides
template<int ndim>
int caterva_copy_buffer(uint8_t itemsize,
                        void *src, int64_t *src_pad_shape,
                        int64_t *src_start, int64_t *src_stop,
                        void *dst, int64_t *dst_pad_shape,
                        int64_t *dst_start) {
    // Compute the shape of the copy
    int64_t copy_shape[ndim];
    for (int i = 0; i < ndim; ++i) {
        copy_shape[i] = src_stop[i] - src_start[i];
        if(copy_shape[i] == 0) {
            return CATERVA_SUCCEED;
        }
    }

    // Compute the strides
    int64_t src_strides[ndim];
    src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        src_strides[i] = src_strides[i + 1] * src_pad_shape[i + 1];
    }

    int64_t dst_strides[ndim];
    dst_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        dst_strides[i] = dst_strides[i + 1] * dst_pad_shape[i + 1];
    }

    // Align the buffers removing unnecessary data
    int64_t src_start_n;
    index_multidim_to_unidim(src_start, ndim, src_strides, &src_start_n);
    uint8_t *bsrc = (uint8_t *) src;
    bsrc = &bsrc[src_start_n * itemsize];

    int64_t dst_start_n;
    index_multidim_to_unidim(dst_start, ndim, dst_strides, &dst_start_n);
    uint8_t *bdst = (uint8_t *) dst;
    bdst = &bdst[dst_start_n * itemsize];

    ndim_copy<ndim>(itemsize, copy_shape, bsrc, src_strides, bdst, dst_strides);
    return CATERVA_SUCCEED;
}

#endif  // CATERVA_CATERVA_UTILS_H_
