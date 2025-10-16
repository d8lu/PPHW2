#include "mpi_sift.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "image.hpp"
#include "sift.hpp"
using namespace std;

// serial version for reference
Image gaussian_blur(const Image& img, float sigma);

struct RowSplit {
    int y0, y1;
};  // half-open [y0, y1)
inline RowSplit split_rows(int H, int P, int r) {
    int base = H / P, rem = H % P;
    int y0 = r * base + std::min(r, rem);
    int cnt = base + (r < rem ? 1 : 0);
    return {y0, y0 + cnt};
}

Image guassian_blur_mpi(const Image& img, float sigma, int rank, int world,
                        MPI_Comm comm, int globalH) {
    // 1. KERNEL CREATION (Identical on all processes)
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;
    std::vector<float> kernel(size);
    float sum = 0;
    for (int k = -size / 2; k <= size / 2; k++) {
        float val = std::exp(-(k * k) / (2 * sigma * sigma));
        kernel[center + k] = val;
        sum += val;
    }
    for (int k = 0; k < size; k++)
        kernel[k] /= sum;

    // Early exit for trivial blur
    if (sigma == 0.f) {
        return img;
    }

    const int W = img.width;
    const int local_h = img.height;
    const auto my_split = split_rows(globalH, world, rank);
    const int my_y0 = my_split.y0;
    const int my_y1 = my_split.y1;

    // 2. BUFFER PREPARATION
    // Create a temporary image to hold this rank's rows plus halo regions.
    Image local_with_halos(W, local_h + 2 * center, 1);
    // Copy this rank's data into the middle of the temporary image
    std::copy(img.data, img.data + W * local_h, local_with_halos.data + center * W);

    // 3. GENERALIZED HALO EXCHANGE
    std::vector<MPI_Request> requests;

    printf("[Rank %d, H=%d] Starting halo exchange (kernel size %d, center %d).\n", rank, globalH, size, center);
    fflush(stdout);
    // Plan communications with all other ranks
    for (int other_rank = 0; other_rank < world; ++other_rank) {
        if (other_rank == rank) continue;

        auto other_split = split_rows(globalH, world, other_rank);
        int other_y0 = other_split.y0;
        int other_y1 = other_split.y1;

        // --- Do I need to RECEIVE from this `other_rank`? ---

        // Check for my TOP halo requirements
        int needed_top_y0 = my_y0 - center;
        int intersect_top_y0 = std::max(needed_top_y0, other_y0);
        int intersect_top_y1 = std::min(my_y0, other_y1);

        if (intersect_top_y0 < intersect_top_y1) {
            int num_rows = intersect_top_y1 - intersect_top_y0;
            printf("[Rank %d] Planning to RECV %d rows for my TOP halo from Rank %d.\n", rank, num_rows, other_rank);
            fflush(stdout);
            int recv_offset_in_halo = intersect_top_y0 - needed_top_y0;
            float* recv_ptr = local_with_halos.data + recv_offset_in_halo * W;
            MPI_Request req;
            MPI_Irecv(recv_ptr, num_rows * W, MPI_FLOAT, other_rank, /*tag=*/0, comm, &req);
            requests.push_back(req);
        }

        // Check for my BOTTOM halo requirements
        int needed_bot_y0 = my_y1;
        int needed_bot_y1 = my_y1 + center;
        int intersect_bot_y0 = std::max(needed_bot_y0, other_y0);
        int intersect_bot_y1 = std::min(needed_bot_y1, other_y1);

        if (intersect_bot_y0 < intersect_bot_y1) {
            int num_rows = intersect_bot_y1 - intersect_bot_y0;
            printf("[Rank %d] Planning to RECV %d rows for my BOTTOM halo from Rank %d.\n", rank, num_rows, other_rank);
            fflush(stdout);

            int recv_offset_in_halo = (local_h + center) + (intersect_bot_y0 - needed_bot_y0);
            float* recv_ptr = local_with_halos.data + recv_offset_in_halo * W;
            MPI_Request req;
            MPI_Irecv(recv_ptr, num_rows * W, MPI_FLOAT, other_rank, /*tag=*/1, comm, &req);
            requests.push_back(req);
        }

        // --- Do I need to SEND to this `other_rank`? ---

        // Check if they need a piece of my data for THEIR top halo
        int other_needed_top_y0 = other_y0 - center;
        int other_intersect_top_y0 = std::max(other_needed_top_y0, my_y0);
        int other_intersect_top_y1 = std::min(other_y0, my_y1);

        if (other_intersect_top_y0 < other_intersect_top_y1) {
            int num_rows = other_intersect_top_y1 - other_intersect_top_y0;
            printf("[Rank %d] Planning to SEND %d rows to Rank %d for THEIR top halo.\n", rank, num_rows, other_rank);
            fflush(stdout);

            int send_offset_in_my_data = other_intersect_top_y0 - my_y0;
            const float* send_ptr = img.data + send_offset_in_my_data * W;
            MPI_Request req;
            MPI_Isend(const_cast<float*>(send_ptr), num_rows * W, MPI_FLOAT, other_rank, /*tag=*/0, comm, &req);
            requests.push_back(req);
        }

        // Check if they need a piece of my data for THEIR bottom halo
        int other_needed_bot_y0 = other_y1;
        int other_needed_bot_y1 = other_y1 + center;
        int other_intersect_bot_y0 = std::max(other_needed_bot_y0, my_y0);
        int other_intersect_bot_y1 = std::min(other_needed_bot_y1, my_y1);

        if (other_intersect_bot_y0 < other_intersect_bot_y1) {
            int num_rows = other_intersect_bot_y1 - other_intersect_bot_y0;
            printf("[Rank %d] Planning to SEND %d rows to Rank %d for THEIR bottom halo.\n", rank, num_rows, other_rank);
            fflush(stdout);

            int send_offset_in_my_data = other_intersect_bot_y0 - my_y0;
            const float* send_ptr = img.data + send_offset_in_my_data * W;
            MPI_Request req;
            MPI_Isend(const_cast<float*>(send_ptr), num_rows * W, MPI_FLOAT, other_rank, /*tag=*/1, comm, &req);
            requests.push_back(req);
        }
    }

    // Execute all planned communications
    if (!requests.empty()) {
        printf("[Rank %d] Entering MPI_Waitall for %zu requests...\n", rank, requests.size());
        fflush(stdout);
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        printf("[Rank %d] Exited MPI_Waitall.\n", rank);
        fflush(stdout);
    }

    // 4. BOUNDARY CONDITIONS (clamp-to-edge)
    // Manually handle parts of the halo that fall off the global image.
    if (rank == 0) {
        const float* first_row_src = img.data;
        for (int i = 0; i < center; ++i) {
            std::copy(first_row_src, first_row_src + W, local_with_halos.data + i * W);
        }
    }
    if (rank == world - 1) {
        const float* last_row_src = img.data + (local_h - 1) * W;
        for (int i = 0; i < center; ++i) {
            std::copy(last_row_src, last_row_src + W, local_with_halos.data + (local_h + center + i) * W);
        }
    }

    // 5. VERTICAL CONVOLUTION
    Image tmp(W, local_h, 1);
    for (int y = 0; y < local_h; y++) {
        for (int x = 0; x < W; x++) {
            float pixel_sum = 0;
            int y_src_base = y + center;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                pixel_sum += local_with_halos.data[(y_src_base + dy) * W + x] * kernel[k];
            }
            tmp.data[y * W + x] = pixel_sum;
        }
    }

    // 6. HORIZONTAL CONVOLUTION (local, no communication needed)
    Image filtered(W, local_h, 1);
    for (int y = 0; y < local_h; y++) {
        for (int x = 0; x < W; x++) {
            float pixel_sum = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                pixel_sum += tmp.get_pixel(x + dx, y, 0) * kernel[k];
            }
            filtered.set_pixel(x, y, 0, pixel_sum);
        }
    }

    return filtered;
}

ScaleSpacePyramid generate_gaussian_pyramid_mpi(const Image& img,
                                                float sigma_min,
                                                int num_octaves,
                                                int scales_per_octave) {
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // ----- Rank 0 builds the base image (2x resize + blur) -----
    Image base_img;
    float base_sigma = 0.f;
    if (rank == 0) {
        base_sigma = sigma_min / MIN_PIX_DIST;
        base_img =
            img.resize(img.width * 2, img.height * 2, Interpolation::BILINEAR);
        float sigma_diff = std::sqrt(std::max(0.f, base_sigma * base_sigma - 1.0f));
        base_img = gaussian_blur(base_img, sigma_diff);
        assert(base_img.channels == 1 &&
               "Expect grayscale (C==1) before MPI scatter.");
    }

    // ----- Broadcast dims + base_sigma so every rank can allocate -----
    int W = 0, H = 0;
    if (rank == 0) {
        W = base_img.width;
        H = base_img.height;
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_sigma, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ----- Compute this rank's row range and scatter the base image -----
    auto my_split_base = split_rows(H, world, rank);
    Image local_rows(W, my_split_base.y1 - my_split_base.y0, 1);

    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(world);
        displs.resize(world);
        for (int r = 0; r < world; ++r) {
            auto s = split_rows(H, world, r);
            counts[r] = (s.y1 - s.y0) * W;
            displs[r] = s.y0 * W;
        }
    }

    MPI_Scatterv(rank == 0 ? base_img.data : nullptr,
                 rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr, MPI_FLOAT,
                 local_rows.data, local_rows.width * local_rows.height,
                 MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ---- Each rank builds its local pyramid stripe ----
    int imgs_per_octave = scales_per_octave + 3;
    float k = std::pow(2, 1.0 / scales_per_octave);
    std::vector<float> sigma_vals(imgs_per_octave);
    // Mimic serial implementation for consistency
    sigma_vals[0] = base_sigma;
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i - 1);
        float sigma_total = k * sigma_prev;
        sigma_vals[i] = std::sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev);
    }

    std::vector<std::vector<Image>> local_octaves(num_octaves);
    Image octave_base = local_rows;
    int current_H = H;
    int current_W = W;

    for (int oi = 0; oi < num_octaves; ++oi) {
        local_octaves[oi].reserve(imgs_per_octave);
        local_octaves[oi].push_back(octave_base);

        // progressively blur
        for (int si = 1; si < imgs_per_octave; ++si) {
            Image blurred =
                guassian_blur_mpi(local_octaves[oi][si - 1], sigma_vals[si],
                                  rank, world, MPI_COMM_WORLD, current_H);
            local_octaves[oi].push_back(std::move(blurred));
        }

        // downsample for next octave
        if (oi + 1 < num_octaves) {
            const Image& local_src = local_octaves[oi][imgs_per_octave - 3];

            Image gathered;
            if (rank == 0) gathered = Image(current_W, current_H, 1);

            if (rank == 0) {
                counts.assign(world, 0);
                displs.assign(world, 0);
                for (int r = 0; r < world; ++r) {
                    auto s = split_rows(current_H, world, r);
                    counts[r] = (s.y1 - s.y0) * current_W;
                    displs[r] = s.y0 * current_W;
                }
            }

            MPI_Gatherv(local_src.data, local_src.width * local_src.height,
                        MPI_FLOAT, rank == 0 ? gathered.data : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displs.data() : nullptr, MPI_FLOAT, 0,
                        MPI_COMM_WORLD);

            // Root downsamples exactly like the serial version
            int Wn = current_W / 2;
            int Hn = current_H / 2;
            Image down_all;
            if (rank == 0) {
                // FIX: The original manual downsampling was incorrect. Using the known-correct
                // resize function from the serial implementation fixes this potential bug.
                down_all = gathered.resize(Wn, Hn, Interpolation::NEAREST);
            }

            // Scatter the new base for next octave
            auto split_n = split_rows(Hn, world, rank);
            Image next_local(Wn, split_n.y1 - split_n.y0, 1);

            if (rank == 0) {
                counts.assign(world, 0);
                displs.assign(world, 0);
                for (int r = 0; r < world; ++r) {
                    auto s = split_rows(Hn, world, r);
                    counts[r] = (s.y1 - s.y0) * Wn;
                    displs[r] = s.y0 * Wn;
                }
            }
            MPI_Scatterv(rank == 0 ? down_all.data : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displs.data() : nullptr, MPI_FLOAT,
                         next_local.data, next_local.width * next_local.height,
                         MPI_FLOAT, 0, MPI_COMM_WORLD);

            octave_base = std::move(next_local);
            current_H = Hn;
            current_W = Wn;
        }
    }

    // ---- GATHER FINAL PYRAMID TO RANK 0 ----
    ScaleSpacePyramid pyramid;
    pyramid.octaves = std::move(local_octaves);

    pyramid.num_octaves = num_octaves;
    pyramid.imgs_per_octave = imgs_per_octave;
    return pyramid;
    // if (rank == 0) {
    //     pyramid.num_octaves = num_octaves;
    //     pyramid.imgs_per_octave = imgs_per_octave;
    //     pyramid.octaves.assign(num_octaves, {});
    // }

    // int H_o = H;
    // int W_o = W;
    // for (int oi = 0; oi < num_octaves; ++oi) {
    //     if (rank == 0) {
    //         pyramid.octaves[oi].resize(imgs_per_octave);
    //          counts.assign(world, 0);
    //          displs.assign(world, 0);
    //         for (int r = 0; r < world; ++r) {
    //             auto s = split_rows(H_o, world, r);
    //             counts[r] = (s.y1 - s.y0) * W_o;
    //             displs[r] = s.y0 * W_o;
    //         }
    //     }

    //     for (int si = 0; si < imgs_per_octave; ++si) {
    //         const Image& local_im = local_octaves[oi][si];
    //         if (rank == 0) pyramid.octaves[oi][si] = Image(W_o, H_o, 1);

    //         MPI_Gatherv(local_im.data, local_im.width * local_im.height,
    //                     MPI_FLOAT,
    //                     rank == 0 ? pyramid.octaves[oi][si].data : nullptr,
    //                     rank == 0 ? counts.data() : nullptr,
    //                     rank == 0 ? displs.data() : nullptr, MPI_FLOAT, 0,
    //                     MPI_COMM_WORLD);
    //     }
    //     H_o /= 2;
    //     W_o /= 2;
    // }

    // if (rank == 0)
    //     return pyramid;
    // else
    //     return ScaleSpacePyramid{0, 0, {}};
}

ScaleSpacePyramid generate_dog_pyramid_mpi(const ScaleSpacePyramid& img_pyramid, int rank, int world_size) {
    return {};
}