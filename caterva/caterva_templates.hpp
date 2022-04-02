#pragma once
#include <blosc2.h>
#include <caterva.h>
#include <caterva_utils_templates.hpp>


// Only for internal use: It is used for getting slices.
template<int ndim>
int caterva_blosc_get_slice(caterva_ctx_t *ctx, void *buffer, int64_t buffersize, int64_t *start,
                        int64_t *stop, int64_t *shape, caterva_array_t *array) {
    CATERVA_ERROR_NULL(ctx);
    CATERVA_ERROR_NULL(buffer);
    CATERVA_ERROR_NULL(start);
    CATERVA_ERROR_NULL(stop);
    CATERVA_ERROR_NULL(array);
    if (buffersize < 0) {
        CATERVA_TRACE_ERROR("buffersize is < 0");
        CATERVA_ERROR(CATERVA_ERR_INVALID_ARGUMENT);
    }
    uint64_t stack_maskout[STACK_MASKOUT_BITS / 64];
    uint8_t *buffer_b = (uint8_t *) buffer;

    // 0-dim case
    if constexpr (ndim == 0) {
        if (blosc2_schunk_decompress_chunk(array->sc, 0, buffer_b, array->itemsize) < 0) {
            CATERVA_ERROR(CATERVA_ERR_BLOSC_FAILED);
        }
        return CATERVA_SUCCEED;
    }

    int32_t data_nbytes = array->extchunknitems * array->itemsize;
    uint8_t *data = (uint8_t*)malloc(data_nbytes);

    int64_t chunks_in_array[CATERVA_MAX_DIM] = {0};
    for (int i = 0; i < ndim; ++i) {
        chunks_in_array[i] = array->extshape[i] / array->chunkshape[i];
    }

    int64_t chunks_in_array_strides[CATERVA_MAX_DIM];
    chunks_in_array_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        chunks_in_array_strides[i] = chunks_in_array_strides[i + 1] * chunks_in_array[i + 1];
    }

    int64_t blocks_in_chunk[CATERVA_MAX_DIM] = {0};
    for (int i = 0; i < ndim; ++i) {
        blocks_in_chunk[i] = array->extchunkshape[i] / array->blockshape[i];
    }

    // Compute the number of chunks to update
    int64_t update_start[CATERVA_MAX_DIM];
    int64_t update_shape[CATERVA_MAX_DIM];

    int64_t update_nchunks = 1;
    for (int i = 0; i < ndim; ++i) {
        // round start down to nearest multiple of chunkshape
        int64_t pos = start[i] - (start[i] % array->chunkshape[i]);
        update_start[i] = pos / array->chunkshape[i];

        // round stop up to nearest multiple of chunkshape
        pos = stop[i] + array->chunkshape[i] - 1 - (stop[i] + array->chunkshape[i] - 1) % array->chunkshape[i];
        update_shape[i] = pos / array->chunkshape[i] - update_start[i];

        update_nchunks *= update_shape[i];
    }

    // number of blocks per chunk
    int64_t blocks_per_chunk = array->extchunknitems / array->blocknitems;
    uint64_t *block_maskout;
    if (blocks_per_chunk <= STACK_MASKOUT_BITS) {  // in most cases we won't need to dynamically allocate
        block_maskout = stack_maskout;
    } else {
        int nmaskoutbits = (blocks_per_chunk + 63) & (-64);  // round up to next multiple of 64
        int nmaskoutelems = nmaskoutbits / 64;
        uint64_t *block_maskout = (uint64_t*)ctx->cfg->alloc(nmaskoutelems * 8);
        CATERVA_ERROR_NULL(block_maskout);
    }

    for (int update_nchunk = 0; update_nchunk < update_nchunks; ++update_nchunk) {
        int64_t nchunk_ndim[CATERVA_MAX_DIM] = {0};
        index_unidim_to_multidim<ndim>(update_shape, update_nchunk, nchunk_ndim);
        for (int i = 0; i < ndim; ++i) {
            nchunk_ndim[i] += update_start[i];
        }
        int64_t nchunk;
        index_multidim_to_unidim<ndim>(nchunk_ndim, chunks_in_array_strides, &nchunk);

        // check if the chunk needs to be updated
        int64_t chunk_start[CATERVA_MAX_DIM] = {0};
        int64_t chunk_stop[CATERVA_MAX_DIM] = {0};
        for (int i = 0; i < ndim; ++i) {
            chunk_start[i] = nchunk_ndim[i] * array->chunkshape[i];
            chunk_stop[i] = chunk_start[i] + array->chunkshape[i];
            if (chunk_stop[i] > array->shape[i]) {
                
                chunk_stop[i] = array->shape[i];
            }
        }
        // bool chunk_empty = false;
        // for (int i = 0; i < ndim; ++i) {
        //     chunk_empty |= ((chunk_stop[i] <= start[i]) | (chunk_start[i] >= stop[i]));
        // }
        // if (chunk_empty) {
        //     continue;
        // }

        uint64_t maskVal = 0;
        int64_t maskout_offset = 0;
        int64_t maskout_index = 0;
        for (int nblock = 0; nblock < blocks_per_chunk; ++nblock) {
            if (maskout_offset == 64) {
                block_maskout[maskout_index] = maskVal;
                maskout_offset = 0;
                maskVal = 0;
                ++maskout_index;
            }

            int64_t nblock_ndim[CATERVA_MAX_DIM] = {0};
            index_unidim_to_multidim<ndim>(blocks_in_chunk, nblock, nblock_ndim);

            // check if the block needs to be updated
            int64_t block_start[CATERVA_MAX_DIM] = {0};
            int64_t block_stop[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                block_start[i] = nblock_ndim[i] * array->blockshape[i];
                block_stop[i] = block_start[i] + array->blockshape[i];
                block_start[i] += chunk_start[i];
                block_stop[i] += chunk_start[i];

                if (block_start[i] > chunk_stop[i]) {
                    block_start[i] = chunk_stop[i];
                }
                if (block_stop[i] > chunk_stop[i]) {
                    block_stop[i] = chunk_stop[i];
                }
            }

            uint64_t block_empty = 0;
            for (int i = 0; i < ndim; ++i) {
                block_empty |= (block_stop[i] <= start[i]) | (block_start[i] >= stop[i]);
            }
            maskVal |= block_empty << maskout_offset;
            ++maskout_offset;
        }

        block_maskout[maskout_index] = maskVal;

        if (blosc2_set_maskout_bitmask(array->sc->dctx, block_maskout, blocks_per_chunk) !=
            BLOSC2_ERROR_SUCCESS) {
            CATERVA_TRACE_ERROR("Error setting the maskout");
            CATERVA_ERROR(CATERVA_ERR_BLOSC_FAILED);
        }

        int err = blosc2_schunk_decompress_chunk(array->sc, nchunk, data, data_nbytes);
        if (err < 0) {
            CATERVA_TRACE_ERROR("Error decompressing chunk");
            CATERVA_ERROR(CATERVA_ERR_BLOSC_FAILED);
        }

        // Iterate over blocks

        for (int nblock = 0; nblock < blocks_per_chunk; ++nblock) {

            // skip known empty blocks
            if (block_maskout[nblock / 64] & (1ULL << (nblock % 64))) {
                continue;
            }

            int64_t nblock_ndim[CATERVA_MAX_DIM] = {0};
            index_unidim_to_multidim<ndim>(blocks_in_chunk, nblock, nblock_ndim);

            // check if the block needs to be updated
            int64_t block_start[CATERVA_MAX_DIM] = {0};
            int64_t block_stop[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                block_start[i] = nblock_ndim[i] * array->blockshape[i];
                block_stop[i] = block_start[i] + array->blockshape[i];
                block_start[i] += chunk_start[i];
                block_stop[i] += chunk_start[i];

                // if (block_start[i] > chunk_stop[i]) {
                //     block_start[i] = chunk_stop[i];
                // }
                if (block_stop[i] > chunk_stop[i]) {
                    block_stop[i] = chunk_stop[i];
                }
            }
            int64_t block_shape[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                block_shape[i] = block_stop[i] - block_start[i];
            }

            // compute the start of the slice inside the block
            int64_t slice_start[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                if (block_start[i] < start[i]) {
                    slice_start[i] = start[i] - block_start[i];
                } else {
                    slice_start[i] = 0;
                }
                slice_start[i] += block_start[i];
            }

            int64_t slice_stop[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                if (block_stop[i] > stop[i]) {
                    slice_stop[i] = block_shape[i] - (block_stop[i] - stop[i]);
                } else {
                    slice_stop[i] = block_stop[i] - block_start[i];
                }
                slice_stop[i] += block_start[i];
            }

            int64_t slice_shape[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                slice_shape[i] = slice_stop[i] - slice_start[i];
            }

            uint8_t *dst = &buffer_b[0];
            int64_t *dst_pad_shape = shape;

            int64_t dst_start[CATERVA_MAX_DIM] = {0};
            //int64_t dst_stop[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                dst_start[i] = slice_start[i] - start[i];
                //dst_stop[i] = slice_stop[i] - start[i];
            }

            uint8_t *src = &data[nblock * array->blocknitems * array->itemsize];
            int64_t src_pad_shape[CATERVA_MAX_DIM];
            for (int i = 0; i < ndim; ++i) {
                src_pad_shape[i] = array->blockshape[i];
            }

            int64_t src_start[CATERVA_MAX_DIM] = {0};
            int64_t src_stop[CATERVA_MAX_DIM] = {0};
            for (int i = 0; i < ndim; ++i) {
                src_start[i] = slice_start[i] - block_start[i];
                src_stop[i] = src_start[i] + slice_shape[i];
            }

            caterva_copy_buffer<ndim>(array->itemsize, src, src_pad_shape, src_start, src_stop,
                                dst, dst_pad_shape, dst_start);
        }
    }
    if (blocks_per_chunk > STACK_MASKOUT_BITS) {
        ctx->cfg->free(block_maskout);
    }

    free(data);

    return CATERVA_SUCCEED;
}

template<int ndim>
int caterva_get_slice_buffer(caterva_ctx_t *ctx,
                             caterva_array_t *array,
                             int64_t *start, int64_t *stop,
                             void *buffer, int64_t *buffershape, int64_t buffersize) {
    CATERVA_ERROR_NULL(ctx);
    CATERVA_ERROR_NULL(array);
    CATERVA_ERROR_NULL(start);
    CATERVA_ERROR_NULL(stop);
    CATERVA_ERROR_NULL(buffershape);
    CATERVA_ERROR_NULL(buffer);

    int64_t size = array->itemsize;
    for (int i = 0; i < ndim; ++i) {
        if (stop[i] - start[i] > buffershape[i]) {
            CATERVA_TRACE_ERROR("The buffer shape can not be smaller than the slice shape");
            return CATERVA_ERR_INVALID_ARGUMENT;
        }
        size *= buffershape[i];
    }

    if (array->nitems == 0) {
        return CATERVA_SUCCEED;
    }

    if (buffersize < size) {
        CATERVA_ERROR(CATERVA_ERR_INVALID_ARGUMENT);
    }
    CATERVA_ERROR(caterva_blosc_get_slice<ndim>(ctx, buffer, buffersize, start, stop, buffershape, array));

    return CATERVA_SUCCEED;
}

template<int ndim>
int caterva_to_buffer(caterva_ctx_t *ctx, caterva_array_t *array, void *buffer,
                      int64_t buffersize) {
    CATERVA_ERROR_NULL(ctx);
    CATERVA_ERROR_NULL(array);
    CATERVA_ERROR_NULL(buffer);

    if (buffersize < (int64_t) array->nitems * array->itemsize) {
        CATERVA_ERROR(CATERVA_ERR_INVALID_ARGUMENT);
    }

    if (array->nitems == 0) {
        return CATERVA_SUCCEED;
    }

    int64_t start[CATERVA_MAX_DIM] = {0};
    int64_t *stop = array->shape;
    CATERVA_ERROR(caterva_get_slice_buffer<ndim>(ctx, array, start, stop,
                                           buffer, array->shape, buffersize));
    return CATERVA_SUCCEED;
}