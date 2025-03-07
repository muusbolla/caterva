/*
 * Copyright (C) 2018 Francesc Alted, Aleix Alcacer.
 * Copyright (C) 2019-present Blosc Development team <blosc@blosc.org>
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 *
 * To get the used frames (air1.cat, precip1.cat, precip2.cat, precip3.cat, snow1.cat...)
 * you must use the following script "fecth_data.py":

import os
import sys
import xarray as xr
import numpy as np
import s3fs
import caterva as cat

def open_zarr(year, month, datestart, dateend):
    fs = s3fs.S3FileSystem(anon=True)
    datestring = "era5-pds/zarr/{year}/{month:02d}/data/".format(year=year, month=month)
    s3map = s3fs.S3Map(datestring + "precipitation_amount_1hour_Accumulation.zarr/", s3=fs)
    precip_zarr = xr.open_dataset(s3map, engine="zarr")
    precip_zarr = precip_zarr.sel(time1=slice(np.datetime64(datestart), np.datetime64(dateend)))

    return precip_zarr.precipitation_amount_1hour_Accumulation

print("Fetching data from S3 (era5-pds)...")
precip_m0 = open_zarr(1987, 10, "1987-10-01", "1987-10-30 23:59")

if os.path.exists("precip1.cat"):
    cat.remove(path)

# ia.set_config_defaults(favor=ia.Favor.SPEED)
m_shape = precip_m0.shape
m_chunks = (128, 128, 256)
m_blocks = (32, 32, 32)
cat_precip0 = cat.empty(m_shape, itemsize=4, chunks=m_chunks, blocks=m_blocks,
                        urlpath="precip1.cat", sequential=True)
print("Fetching and storing 1st month...")
values = precip_m0.values
cat_precip0[:] = values

 * To call this script, you can run the following commands:
 * pip install caterva
 * python fetch_data.py
 *
 */

# include <caterva.h>
# include "../contribs/c-blosc2/include/blosc2/codecs-registry.h"
# include "../contribs/c-blosc2/plugins/codecs/zfp/blosc2-zfp.h"

int comp(const char* urlpath) {
    blosc_init();

    blosc2_schunk *schunk = blosc2_schunk_open(urlpath);

    if (schunk->typesize != 4) {
        printf("Error: This test is only for floats.\n");
        return -1;
    }

    blosc2_remove_urlpath("schunk_rate.cat");
    blosc2_remove_urlpath("schunk.cat");

    // Get multidimensional parameters and configure Caterva array
    uint8_t ndim;
    int32_t shape[4];
    int64_t *shape_aux = malloc(8 * sizeof(int64_t));
    int32_t *chunkshape = malloc(8 * sizeof(int32_t));
    int32_t *blockshape = malloc(8 * sizeof(int32_t));
    uint8_t *smeta;
    uint32_t smeta_len;
    if (blosc2_meta_get(schunk, "caterva", &smeta, &smeta_len) < 0) {
        printf("This benchmark only supports Caterva datasets");
        free(shape);
        free(chunkshape);
        free(blockshape);
        return -1;
    }
    deserialize_meta(smeta, smeta_len, &ndim, shape_aux, chunkshape, blockshape);
    free(smeta);

    caterva_config_t cfg = CATERVA_CONFIG_DEFAULTS;
    cfg.nthreads = 6;
    caterva_ctx_t *ctx, *ctx_zfp;
    caterva_ctx_new(&cfg, &ctx);

    caterva_storage_t storage = {0};
    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = chunkshape[i];
        storage.blockshape[i] = blockshape[i];
        shape[i] = (int32_t) shape_aux[i];
    }

    caterva_array_t *arr;
    caterva_from_schunk(ctx, schunk, &arr);
    int copied;
    printf("LZ4 comp ratio: %f \n",(float) arr->sc->nbytes / (float) arr->sc->cbytes);

    /* Use BLOSC_CODEC_ZFP_FIXED_RATE */
    storage.urlpath = "schunk_rate.cat";
    caterva_array_t *arr_rate;
    ctx_zfp = ctx;
    ctx_zfp->cfg->compcodec = BLOSC_CODEC_ZFP_FIXED_RATE;
    ctx_zfp->cfg->splitmode = BLOSC_NEVER_SPLIT;
    ctx_zfp->cfg->compmeta = (uint8_t) (100.0 * (float) arr->sc->cbytes / (float) arr->sc->nbytes);
    ctx_zfp->cfg->filters[BLOSC2_MAX_FILTERS - 1] = 0;
    ctx_zfp->cfg->filtersmeta[BLOSC2_MAX_FILTERS - 1] = 0;
    copied = caterva_copy(ctx_zfp, arr, &storage, &arr_rate);
    if (copied != 0) {
        printf("Error BLOSC_CODEC_ZFP_FIXED_RATE \n");
        free(shape);
        free(chunkshape);
        free(blockshape);
        caterva_free(ctx_zfp, &arr);
        return -1;
    }
    printf("ZFP_FIXED_RATE comp ratio: %f \n",(float) arr_rate->sc->nbytes / (float) arr_rate->sc->cbytes);

    int nelems = arr_rate->nitems;
    int index, dsize_zfp, dsize_blosc;
    float item_zfp, item_blosc;
    blosc_timestamp_t t0, t1;
    double zfp_time, blosc_time;
    zfp_time = blosc_time = 0;
    int64_t index_ndim[ZFP_MAX_DIM];
    int64_t index_chunk_ndim[ZFP_MAX_DIM];
    int64_t ind_ndim[ZFP_MAX_DIM];
    int stride_chunk, nchunk, ind_chunk;
    bool needs_free_blosc, needs_free_zfp;
    uint8_t *chunk_blosc, *chunk_zfp;
    int32_t chunk_nbytes_zfp, chunk_cbytes_zfp, chunk_nbytes_lossy, chunk_cbytes_lossy;
    double ntests = 500.0;
    for (int i = 0; i < ntests; ++i) {
        srand(i);
        index = rand() % nelems;
        index_unidim_to_multidim(ndim, shape, index, index_ndim);
        for (int j = 0; j < ndim; ++j) {
            index_chunk_ndim[j] = index_ndim[j] / chunkshape[j];
            ind_ndim[j] = index_ndim[j] % chunkshape[j];
        }
        stride_chunk = (shape[1] - 1) / chunkshape[1] + 1;
        nchunk = index_chunk_ndim[0] * stride_chunk + index_chunk_ndim[1];
        ind_chunk = ind_ndim[0] * chunkshape[1] + ind_ndim[1];
        blosc2_schunk_get_lazychunk(arr->sc, nchunk, &chunk_blosc, &needs_free_blosc);
        blosc2_cbuffer_sizes(chunk_blosc, &chunk_nbytes_lossy, &chunk_cbytes_lossy, NULL);
        blosc_set_timestamp(&t0);
        dsize_blosc = blosc2_getitem_ctx(arr->sc->dctx, chunk_blosc, chunk_cbytes_lossy,
                                       ind_chunk, 1, &item_blosc, sizeof(item_blosc));
        blosc_set_timestamp(&t1);
        blosc_time += blosc_elapsed_secs(t0, t1);
        blosc2_schunk_get_lazychunk(arr_rate->sc, nchunk, &chunk_zfp, &needs_free_zfp);
        blosc2_cbuffer_sizes(chunk_zfp, &chunk_nbytes_zfp, &chunk_cbytes_zfp, NULL);
        blosc_set_timestamp(&t0);
        dsize_zfp = blosc2_getitem_ctx(arr_rate->sc->dctx, chunk_zfp, chunk_cbytes_zfp,
                                         ind_chunk, 1, &item_zfp, sizeof(item_zfp));
        blosc_set_timestamp(&t1);
        zfp_time += blosc_elapsed_secs(t0, t1);
        if (dsize_blosc != dsize_zfp) {
            printf("Different amount of items gotten");
            return -1;
        }
    }
    printf("ZFP_FIXED_RATE time: %.5f microseconds\n", (zfp_time * 1000000.0 / ntests));
    printf("Blosc2 time: %.5f microseconds\n", (blosc_time * 1000000.0 / ntests));

    free(shape_aux);
    free(chunkshape);
    free(blockshape);
    caterva_free(ctx_zfp, &arr);
    caterva_free(ctx_zfp, &arr_rate);
    caterva_ctx_free(&ctx_zfp);
    if (needs_free_blosc) {
        free(chunk_blosc);
    }
    if (needs_free_zfp) {
        free(chunk_zfp);
    }
    blosc_destroy();

    return CATERVA_SUCCEED;
}

int solar1() {
    const char* urlpath = "../../bench/solar1.cat";

    int result = comp(urlpath);
    return result;
}

int air1() {
    const char* urlpath = "../../bench/air1.cat";

    int result = comp(urlpath);
    return result;
}

int snow1() {
    const char* urlpath = "../../bench/snow1.cat";

    int result = comp(urlpath);
    return result;
}

int wind1() {
    const char* urlpath = "../../bench/wind1.cat";

    int result = comp(urlpath);
    return result;
}

int precip1() {
    const char* urlpath = "../../bench/precip1.cat";

    int result = comp(urlpath);
    return result;
}

int precip2() {
    const char* urlpath = "../../bench/precip2.cat";

    int result = comp(urlpath);
    return result;
}

int precip3() {
    const char* urlpath = "../../bench/precip3.cat";

    int result = comp(urlpath);
    return result;
}

int precip3m() {
    const char* urlpath = "../../bench/precip-3m.cat";

    int result = comp(urlpath);
    return result;
}


int main() {

    printf("wind1 \n");
    CATERVA_ERROR(wind1());
    printf("air1 \n");
    CATERVA_ERROR(air1());
    printf("solar1 \n");
    CATERVA_ERROR(solar1());
    printf("snow1 \n");
    CATERVA_ERROR(snow1());
    printf("precip1 \n");
    CATERVA_ERROR(precip1());
      printf("precip2 \n");
    CATERVA_ERROR(precip2());
    printf("precip3 \n");
    CATERVA_ERROR(precip3());
//    printf("precip3m \n");
  //  CATERVA_ERROR(precip3m());
    return CATERVA_SUCCEED;

   }
