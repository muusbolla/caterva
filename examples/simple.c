/*
 * Copyright (C) 2018  Francesc Alted
 * Copyright (C) 2018  Aleix Alcacer
 */

#include <caterva.h>

int main(int argc, char **argv){

    // Create a context
    caterva_ctx *ctx = caterva_new_ctx(NULL, NULL, BLOSC_CPARAMS_DEFAULTS, BLOSC_DPARAMS_DEFAULTS);
    ctx->cparams.typesize = sizeof(double);

    // Define the pshape for the first array
    size_t ndim = 3;
    size_t pshape_[] = {3, 2, 4};
    caterva_dims pshape = caterva_new_dims(pshape_, ndim);

    // Create the first array (empty)
    caterva_array *cat1 = caterva_empty_array(ctx, NULL, pshape);

    // Define a buffer shape to fill cat1
    size_t shape_[] = {10, 10, 10};
    caterva_dims shape = caterva_new_dims(shape_, ndim);

    // Create a buffer to fill cat1 and empty it with an arange
    size_t buf1size = 1;
    for (int i = 0; i < shape.ndim; ++i) {
        buf1size *= shape.dims[i];
    }
    double *buf1 = (double *) malloc(buf1size * sizeof(double));

    for (size_t k = 0; k < buf1size; ++k) {
        buf1[k] = (double) k;
    }

    // Fill cat1 with the above buffer
    caterva_from_buffer(cat1, shape, buf1);

    // Apply a `get_slice` to cat1 and store it into cat2
    size_t start_[] = {3, 6, 4};
    caterva_dims start = caterva_new_dims(start_, ndim);
    size_t stop_[] = {4, 9, 8};
    caterva_dims stop = caterva_new_dims(stop_, ndim);

    size_t pshape2_[]  = {1, 2, 3};
    caterva_dims pshape2 = caterva_new_dims(pshape2_, ndim);
    caterva_array *cat2 = caterva_empty_array(ctx, NULL, pshape2);

    caterva_get_slice(cat2, cat1, start, stop);

    // Assert that the `squeeze` works well
    if (cat1->ndim == cat2->ndim) {
        return -1;
    }

    // Create a buffer to store the cat2 elements
    size_t buf2size = 1;
    caterva_dims shape2 = caterva_get_shape(cat2);
    for (int j = 0; j < shape2.ndim; ++j) {
        buf2size *= shape2.dims[j];
    }
    double *buf2 = (double *) malloc(buf2size * sizeof(double));

    // Fill buffer with the cat2 content
    caterva_to_buffer(cat2, buf2);

    // Print results
    printf("The resulting hyperplane is:\n");

    for (int i = 0; i < shape2.dims[0]; ++i) {
        for (int j = 0; j < shape2.dims[1]; ++j) {
            printf("%6.f", buf2[i * cat2->shape[1] + j]);
        }
        printf("\n");
    }

    free(buf1);
    free(buf2);
    return 0;
}