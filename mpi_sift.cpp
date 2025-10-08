#include <mpi.h>
#include "sift.hpp"
#include "image.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>
#include "mpi_sift.hpp"
using namespace std;

Image guassian_blur_mpi(const Image &img, float sigma, int rank, int world, MPI_Comm comm) {
    int size = ceil(6 * sigma);
    if(size%2 == 0) size++;
    int center = size/2;
    Image kernel(size, 1, 1);
    float sum = 0;
        for (int k = -size / 2; k <= size / 2; k++) {
        float val = std::exp(-(k * k) / (2 * sigma * sigma));
        kernel.set_pixel(center + k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++) kernel.data[k] /= sum;

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank + 1) < world ? rank + 1 : MPI_PROC_NULL;
    int center = size / 2;
}
ScaleSpacePyramid generate_gaussian_pyramid_mpi(const Image& img,
                                                float sigma_min,
                                                int num_octaves,
                                                int scales_per_octave)
{
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // ----- Rank 0 builds the base image (2x resize + blur) -----
    Image base_img;                // valid only on rank 0
    float base_sigma = 0.f;        // will be broadcast to all ranks
    if (rank == 0) {
        base_sigma = sigma_min / MIN_PIX_DIST;
        base_img   = img.resize(img.width * 2, img.height * 2,
                                Interpolation::BILINEAR);
        float sigma_diff = std::sqrt(base_sigma * base_sigma - 1.0f);
        base_img = gaussian_blur(base_img, sigma_diff);
        assert(base_img.channels == 1 && "Expect grayscale (C==1) before MPI scatter.");
    }

    // ----- Broadcast dims + base_sigma so every rank can allocate -----
    int W=0, H=0, C=1;
    if (rank == 0) {
         W = base_img.width; H = base_img.height; C = base_img.channels;
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_sigma, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // (void)C; // silence if asserts are off

    // ----- Compute this rank's row range (block + remainder) -----
    int base = H / world, rem = H % world;
    int y0   = rank * base + std::min(rank, rem);
    int cnt  = base + (rank < rem ? 1 : 0);
    int local_h = cnt;

    // ----- Allocate local receive stripe -----
    Image local_rows(W, local_h, 1);

    // ----- Build counts/displs on root (in floats), then scatterv -----
    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(world);
        displs.resize(world);
        for (int r = 0; r < world; ++r) {
            int b  = H / world, m = H % world;
            int s0 = r * b + std::min(r, m);
            int sc = b + (r < m ? 1 : 0);
            counts[r] = sc * W;       // number of floats to send
            displs[r] = s0 * W;       // starting float index
        }
    }

    MPI_Scatterv(
        rank == 0 ? base_img.data : nullptr,               // sendbuf (root only)
        rank == 0 ? counts.data()  : nullptr,              // sendcounts
        rank == 0 ? displs.data()  : nullptr,              // displs
        MPI_FLOAT,
        local_rows.data,                                   // recvbuf
        local_rows.width * local_rows.height,              // recvcount (floats)
        MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    // FINISH DISTRBIUTION
    int imgs_per_octave = scales_per_octave + 3;
    float k = std::pow(2.f, 1.f / scales_per_octave);
    std::vector<float> sigma_vals(imgs_per_octave);
    sigma_vals[0] = base_sigma;
    for (int i = 1; i < imgs_per_octave; ++i) {
        float po = std::pow(k, i - 1);
        sigma_vals[i] = std::sqrt(base_sigma * po * base_sigma * po * (k*k - 1.0f));
    }
}   

