#pragma once
#include "sift.hpp"   // for ScaleSpacePyramid
#include "image.hpp"
#include <mpi.h>

ScaleSpacePyramid generate_gaussian_pyramid_mpi(
    const Image& img,
    float sigma_min,
    int num_octaves,
    int scales_per_octave);

Image guassian_blur_mpi(const Image& img, float sigma, int rank, int world,
                        MPI_Comm comm, int globalH);