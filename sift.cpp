#define _USE_MATH_DEFINES
#include "sift.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

#include "image.hpp"
#include "mpi_sift.hpp"
using namespace std;

auto total_orientation_time = std::chrono::duration<double, std::milli>::zero();
auto total_descriptor_time = std::chrono::duration<double, std::milli>::zero();
struct RowSplit {
    int y0, y1;
};  // half-open [y0, y1)
inline RowSplit split_rows(int H, int P, int r) {
    int base = H / P, rem = H % P;
    int y0 = r * base + std::min(r, rem);
    int cnt = base + (r < rem ? 1 : 0);
    return {y0, y0 + cnt};
}
ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves,
                                            int scales_per_octave) {
    auto sift_start = std::chrono::high_resolution_clock::now();

    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img =
        img.resize(img.width * 2, img.height * 2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma * base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = std::pow(2, 1.0 / scales_per_octave);
    std::vector<float> sigma_vals{base_sigma};
    sigma_vals.resize(imgs_per_octave);
    // cout << imgs_per_octave << endl;
    for (int i = 1; i < imgs_per_octave; i++) {
        float po = std::pow(k, i - 1);
        sigma_vals[i] =
            std::sqrt(base_sigma * po * base_sigma * po * (k * k - 1.0));
    }
    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {num_octaves, imgs_per_octave,
                                 std::vector<std::vector<Image>>(num_octaves)};
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        auto a_start = std::chrono::high_resolution_clock::now();

        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();

            pyramid.octaves[i].push_back(
                gaussian_blur(prev_img, sigma_vals[j]));
            auto a_end = std::chrono::high_resolution_clock::now();
        }
        auto a_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> sift_duration =
            a_end - a_start;

        std::cout << "a time " << sift_duration.count() << std::endl;
        // prepare base image for next octave
        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave - 3];
        base_img = next_base_img.resize(next_base_img.width / 2,
                                        next_base_img.height / 2,
                                        Interpolation::NEAREST);
    }
    auto sift_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> sift_duration =
        sift_end - sift_start;

    std::cout << "generated_gaussian_pyramid: " << sift_duration.count()
              << " ms\n";
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid) {
    for (int i = 0; i < img_pyramid.num_octaves; i++) {
        for (int j = 0; j < img_pyramid.imgs_per_octave; j++) {
            cout << img_pyramid.octaves[i][j].size << " ";
        }
        cout << endl;
    }
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves, img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)};
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            Image diff = img_pyramid.octaves[i][j];
#pragma omp parallel for schedule(static)
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -=
                    img_pyramid.octaves[i][j - 1].data[pix_idx];
            }
            dog_pyramid.octaves[i].push_back(diff);
        }
    }
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y,
                       int rank, int world_size, int W, int H,
                       const float* prev_top_halo, const float* curr_top_halo, const float* next_top_halo,
                       const float* prev_bot_halo, const float* curr_bot_halo, const float* next_bot_halo) {
    float val = octave[scale].get_pixel(x, y, 0);
    bool is_max = true;
    bool is_min = true;

    // Loop over the 26 neighbors in the 3x3x3 cube (3 scales, 3 rows, 3 columns)
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                // Skip comparing the point to itself
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }

                float neighbor_val;
                int ny = y + dy;
                int nx = x + dx;

                // Select the correct image and halo buffers based on the scale offset (dz)
                const Image* neighbor_img_ptr;
                const float* top_halo_ptr;
                const float* bot_halo_ptr;

                if (dz == -1) {  // Previous scale
                    neighbor_img_ptr = &octave[scale - 1];
                    top_halo_ptr = prev_top_halo;
                    bot_halo_ptr = prev_bot_halo;
                } else if (dz == 0) {  // Current scale
                    neighbor_img_ptr = &octave[scale];
                    top_halo_ptr = curr_top_halo;
                    bot_halo_ptr = curr_bot_halo;
                } else {  // Next scale (dz == 1)
                    neighbor_img_ptr = &octave[scale + 1];
                    top_halo_ptr = next_top_halo;
                    bot_halo_ptr = next_bot_halo;
                }

                // --- This is the core logic for using the halos ---
                if (ny < 0) {
                    // This neighbor is in the process above.
                    // If we're rank 0, we're on the global edge, so it can't be an extremum.
                    if (rank == 0) return false;
                    neighbor_val = top_halo_ptr[nx];
                } else if (ny >= H) {
                    // This neighbor is in the process below.
                    // If we're the last rank, we're on the global edge.
                    if (rank == world_size - 1) return false;
                    neighbor_val = bot_halo_ptr[nx];
                } else {
                    // This neighbor is an interior point within this process's chunk.
                    neighbor_val = neighbor_img_ptr->get_pixel(nx, ny, 0);
                }

                // Update flags and check for early exit
                if (neighbor_val > val) is_max = false;
                if (neighbor_val < val) is_min = false;

                if (!is_min && !is_max) {
                    return false;  // Optimization: exit as soon as we know it's not an extremum
                }
            }
        }
    }

    // If the loops complete, the point is an extremum if it was either a pure
    // minimum or a pure maximum relative to all its neighbors.
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale) {
    const Image& img = octave[scale];
    const Image& prev = octave[scale - 1];
    const Image& next = octave[scale + 1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x + 1, y, 0) - img.get_pixel(x - 1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y + 1, 0) - img.get_pixel(x, y - 1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) -
          2 * img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x + 1, y, 0) + img.get_pixel(x - 1, y, 0) -
          2 * img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y + 1, 0) + img.get_pixel(x, y - 1, 0) -
          2 * img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x + 1, y, 0) - next.get_pixel(x - 1, y, 0) -
           prev.get_pixel(x + 1, y, 0) + prev.get_pixel(x - 1, y, 0)) *
          0.25;
    h13 = (next.get_pixel(x, y + 1, 0) - next.get_pixel(x, y - 1, 0) -
           prev.get_pixel(x, y + 1, 0) + prev.get_pixel(x, y - 1, 0)) *
          0.25;
    h23 = (img.get_pixel(x + 1, y + 1, 0) - img.get_pixel(x + 1, y - 1, 0) -
           img.get_pixel(x - 1, y + 1, 0) + img.get_pixel(x - 1, y - 1, 0)) *
          0.25;

    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11 * h22 * h33 - h11 * h23 * h23 - h12 * h12 * h33 +
                2 * h12 * h13 * h23 - h13 * h13 * h22;
    hinv11 = (h22 * h33 - h23 * h23) / det;
    hinv12 = (h13 * h23 - h12 * h33) / det;
    hinv13 = (h12 * h23 - h13 * h22) / det;
    hinv22 = (h11 * h33 - h13 * h13) / det;
    hinv23 = (h12 * h13 - h11 * h23) / det;
    hinv33 = (h11 * h22 - h12 * h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11 * g1 - hinv12 * g2 - hinv13 * g3;
    float offset_x = -hinv12 * g1 - hinv22 * g2 - hinv23 * g3;
    float offset_y = -hinv13 * g1 - hinv23 * g3 - hinv33 * g3;

    float interpolated_extrema_val =
        img.get_pixel(x, y, 0) +
        0.5 * (g1 * offset_s + g2 * offset_x + g3 * offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave,
                      float edge_thresh = C_EDGE) {
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x + 1, y, 0) + img.get_pixel(x - 1, y, 0) -
          2 * img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y + 1, 0) + img.get_pixel(x, y - 1, 0) -
          2 * img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x + 1, y + 1, 0) - img.get_pixel(x + 1, y - 1, 0) -
           img.get_pixel(x - 1, y + 1, 0) + img.get_pixel(x - 1, y - 1, 0)) *
          0.25;

    float det_hessian = h11 * h22 - h12 * h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian * tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh + 1, 2) / edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x,
                           float offset_y, float sigma_min = SIGMA_MIN,
                           float min_pix_dist = MIN_PIX_DIST,
                           int n_spo = N_SPO) {
    kp.sigma = std::pow(2, kp.octave) * sigma_min *
               std::pow(2, (offset_s + kp.scale) / n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x + kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y + kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh) {
    int k = 0;
    bool kp_is_valid = false;
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] =
            fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max(
            {std::abs(offset_s), std::abs(offset_x), std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size() - 1 || kp.scale < 1) break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast &&
            !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid,
                                     float contrast_thresh, float edge_thresh) {
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    std::vector<Keypoint> keypoints;
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];

        for (int j = 1; j < dog_pyramid.imgs_per_octave - 1; j++) {
            const Image& curr_img = octave[j];
            const Image& prev_img = octave[j - 1];
            const Image& next_img = octave[j + 1];
            const int H = curr_img.height;
            const int W = curr_img.width;
            const int PACKED_SIZE = 3 * W;
            vector<float> send_top(PACKED_SIZE);
            vector<float> send_bot(PACKED_SIZE);
            memcpy(send_top.data(), prev_img.data, W * sizeof(float));
            memcpy(send_top.data() + W, curr_img.data, W * sizeof(float));
            memcpy(send_top.data() + 2 * W, next_img.data, W * sizeof(float));

            memcpy(send_bot.data(), prev_img.data + (H - 1) * W, W * sizeof(float));
            memcpy(send_bot.data() + W, curr_img.data + (H - 1) * W, W * sizeof(float));
            memcpy(send_bot.data() + 2 * W, next_img.data + (H - 1) * W, W * sizeof(float));

            vector<float> rec_top(PACKED_SIZE);
            vector<float> rec_bot(PACKED_SIZE);
            int top_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
            int bottom_neighbor = (rank < world - 1) ? rank + 1 : MPI_PROC_NULL;

            MPI_Sendrecv(send_top.data(), PACKED_SIZE, MPI_FLOAT, top_neighbor, 0,
                         rec_bot.data(), PACKED_SIZE, MPI_FLOAT, bottom_neighbor, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Exchange bottom halos (send my packed bottom rows, receive their top rows)
            MPI_Sendrecv(send_bot.data(), PACKED_SIZE, MPI_FLOAT, bottom_neighbor, 1,
                         rec_top.data(), PACKED_SIZE, MPI_FLOAT, top_neighbor, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const float* prev_top_halo = rec_top.data();
            const float* curr_top_halo = rec_top.data() + W;
            const float* next_top_halo = rec_top.data() + 2 * W;
            const float* prev_bot_halo = rec_bot.data();
            const float* curr_bot_halo = rec_bot.data() + W;
            const float* next_bot_halo = rec_bot.data() + 2 * W;

            // #pragma omp parallel
            // {
            std::vector<Keypoint> local;
            // #pragma omp for nowait schedule(static)
            for (int x = 1; x < curr_img.width - 1; x++) {
                for (int y = (rank == 0 ? 1 : 0); y < curr_img.height - (rank == world - 1 ? 1 : 0); y++) {
                    if (std::abs(curr_img.get_pixel(x, y, 0)) <
                        0.8 * contrast_thresh) {
                        continue;
                    }
                    if (point_is_extremum(octave, j, x, y, rank, world, W, H, prev_top_halo, curr_top_halo, next_top_halo, prev_bot_halo, curr_bot_halo, next_bot_halo)) {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(
                            kp, octave, contrast_thresh, edge_thresh);
                        if (kp_is_valid) {
                            local.push_back(kp);
                        }
                    }
                }
            }
            // #pragma omp critical
            keypoints.insert(keypoints.end(), local.begin(), local.end());
            // }
        }
    }
    return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid) {
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            cout << pyramid.octaves[i][j].size << ' ';
        }
        cout << endl;
    }
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves, pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)};

    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].reserve(pyramid.imgs_per_octave);
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            const Image& source_stripe = pyramid.octaves[i][j];
            const int W = source_stripe.width;
            const int H = source_stripe.height;

            vector<float> top_ghost_row(W, 0.0f);
            vector<float> bottom_ghost_row(W, 0.0f);
            int top_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
            int bottom_neighbor = (rank < world - 1) ? rank + 1 : MPI_PROC_NULL;

            // Send UP, receive FROM BELOW into the bottom ghost buffer
            MPI_Sendrecv(source_stripe.data, W, MPI_FLOAT, top_neighbor, 0,
                         bottom_ghost_row.data(), W, MPI_FLOAT, bottom_neighbor, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send DOWN, receive FROM ABOVE into the top ghost buffer
            const float* my_last_row = source_stripe.data + (H - 1) * W;
            MPI_Sendrecv(my_last_row, W, MPI_FLOAT, bottom_neighbor, 1,
                         top_ghost_row.data(), W, MPI_FLOAT, top_neighbor, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Image grad_stripe(W, H, 2);  // Initialized to zeros

// --- FIX #2: Loop Bounds to Match Serial ---
#pragma omp parallel for collapse(2)
            for (int x = 1; x < W - 1; ++x) {
                // This logic correctly skips the global image boundaries
                for (int y = (rank == 0 ? 1 : 0); y < (rank == world - 1 ? H - 1 : H); ++y) {
                    float gx, gy;
                    gx = (source_stripe.get_pixel(x + 1, y, 0) -
                          source_stripe.get_pixel(x - 1, y, 0)) *
                         0.5f;

                    // The internal logic now correctly uses the repaired halo data
                    if (y == 0) {  // Only true for ranks > 0
                        gy = (source_stripe.get_pixel(x, y + 1, 0) - top_ghost_row[x]) * 0.5f;
                    } else if (y == H - 1) {  // Only true for ranks < world-1
                        gy = (bottom_ghost_row[x] - source_stripe.get_pixel(x, y - 1, 0)) * 0.5f;
                    } else {  // Interior pixels for all ranks
                        gy = (source_stripe.get_pixel(x, y + 1, 0) -
                              source_stripe.get_pixel(x, y - 1, 0)) *
                             0.5f;
                    }

                    grad_stripe.set_pixel(x, y, 0, gx);
                    grad_stripe.set_pixel(x, y, 1, gy);
                }
            }
            grad_pyramid.octaves[i].push_back(grad_stripe);
        }
    }
    return grad_pyramid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS]) {
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j - 1 + N_BINS) % N_BINS;
            int next_idx = (j + 1) % N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(
    Keypoint& kp, const ScaleSpacePyramid& grad_pyramid, float lambda_ori,
    float lambda_desc) {
    auto sift_start = std::chrono::high_resolution_clock::now();

    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders
    float min_dist_from_border =
        std::min({kp.x, kp.y, pix_dist * img_grad.width - kp.x,
                  pix_dist * img_grad.height - kp.y});
    if (min_dist_from_border <= std::sqrt(2) * lambda_desc * kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius) / pix_dist);
    int x_end = std::round((kp.x + patch_radius) / pix_dist);
    int y_start = std::round((kp.y - patch_radius) / pix_dist);
    int y_end = std::round((kp.y + patch_radius) / pix_dist);

    // accumulate gradients in orientation histogram
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx * gx + gy * gy);
            weight = std::exp(-(std::pow(x * pix_dist - kp.x, 2) +
                                std::pow(y * pix_dist - kp.y, 2)) /
                              (2 * patch_sigma * patch_sigma));
            theta = std::fmod(std::atan2(gy, gx) + 2 * M_PI, 2 * M_PI);
            bin = (int)std::round(N_BINS / (2 * M_PI) * theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j - 1 + N_BINS) % N_BINS],
                  next = hist[(j + 1) % N_BINS];
            if (prev > hist[j] || next > hist[j]) continue;
            float theta =
                2 * M_PI * (j + 1) / N_BINS +
                M_PI / N_BINS * (prev - next) / (prev - 2 * hist[j] + next);
            orientations.push_back(theta);
        }
    }
    auto sift_end = std::chrono::high_resolution_clock::now();
    total_orientation_time += (sift_end - sift_start);

    return orientations;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc) {
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i - (1 + (float)N_HIST) / 2) * 2 * lambda_desc / N_HIST;
        if (std::abs(x_i - x) > 2 * lambda_desc / N_HIST) continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j - (1 + (float)N_HIST) / 2) * 2 * lambda_desc / N_HIST;
            if (std::abs(y_j - y) > 2 * lambda_desc / N_HIST) continue;

            float hist_weight =
                (1 - N_HIST * 0.5 / lambda_desc * std::abs(x_i - x)) *
                (1 - N_HIST * 0.5 / lambda_desc * std::abs(y_j - y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2 * M_PI * (k - 1) / N_ORI;
                float theta_diff =
                    std::fmod(theta_k - theta_mn + 2 * M_PI, 2 * M_PI);
                if (std::abs(theta_diff) >= 2 * M_PI / N_ORI) continue;
                float bin_weight =
                    1 - N_ORI * 0.5 / M_PI * std::abs(theta_diff);
                hist[i - 1][j - 1][k - 1] += hist_weight * bin_weight * contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI],
                  std::array<uint8_t, 128>& feature_vec) {
    int size = N_HIST * N_HIST * N_ORI;
    float* hist = reinterpret_cast<float*>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], 0.2f * norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++) {
        float val = std::floor(512 * hist[i] / norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc) {
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    // find start and end coords for loops over image patch
    float half_size =
        std::sqrt(2) * lambda_desc * kp.sigma * (N_HIST + 1.) / N_HIST;
    int x_start = std::round((kp.x - half_size) / pix_dist);
    int x_end = std::round((kp.x + half_size) / pix_dist);
    int y_start = std::round((kp.y - half_size) / pix_dist);
    int y_end = std::round((kp.y + half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    // accumulate samples into histograms
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference
            // orientation
            float x = ((m * pix_dist - kp.x) * cos_t +
                       (n * pix_dist - kp.y) * sin_t) /
                      kp.sigma;
            float y = (-(m * pix_dist - kp.x) * sin_t +
                       (n * pix_dist - kp.y) * cos_t) /
                      kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) >
                lambda_desc * (N_HIST + 1.) / N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0),
                  gy = img_grad.get_pixel(m, n, 1);
            float theta_mn =
                std::fmod(std::atan2(gy, gx) - theta + 4 * M_PI, 2 * M_PI);
            float grad_norm = std::sqrt(gx * gx + gy * gy);
            float weight = std::exp(-(std::pow(m * pix_dist - kp.x, 2) +
                                      std::pow(n * pix_dist - kp.y, 2)) /
                                    (2 * patch_sigma * patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn,
                              lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}

// Place this helper function somewhere in your sift.cpp file.
void create_mpi_keypoint_type(MPI_Datatype* mpi_keypoint_type) {
    // A Keypoint contains 4 floats, 4 ints, and 128 uint8_t's
    const int nitems = 3;

    // Setup the description of the struct's members
    int blocklengths[nitems] = {4, 4, 128};
    MPI_Datatype types[nitems] = {MPI_FLOAT, MPI_INT, MPI_UINT8_T};
    MPI_Aint offsets[nitems];

    // Create a dummy instance to calculate memory offsets
    Keypoint kp_dummy;
    MPI_Aint base_address;
    MPI_Get_address(&kp_dummy, &base_address);
    MPI_Get_address(&kp_dummy.x, &offsets[0]);
    MPI_Get_address(&kp_dummy.i, &offsets[1]);
    MPI_Get_address(&kp_dummy.descriptor, &offsets[2]);

    // Make offsets relative to the start of the struct
    for (int i = 0; i < nitems; ++i) {
        offsets[i] -= base_address;
    }

    // Create and commit the custom MPI datatype
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_keypoint_type);
    MPI_Type_commit(mpi_keypoint_type);
}

void exchange_ghost_rows(
    const std::vector<float>& local_data,
    std::vector<float>& top_ghost_buffer,
    std::vector<float>& bottom_ghost_buffer,
    int rows_to_exchange, int width, int local_height, int channels,
    int rank, int world_size) {

    if (rows_to_exchange <= 0) return;

    int top_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom_neighbor = (rank < world_size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    int buffer_size = rows_to_exchange * width * channels;

    // We don't need to resize the buffers here, we assume the caller did it.

    // Exchange with top neighbor
    MPI_Sendrecv(
        &local_data[0],                                    // Send my top rows
        buffer_size, MPI_FLOAT, top_neighbor, 0,
        top_ghost_buffer.data(),                           // Receive into my top ghost buffer
        buffer_size, MPI_FLOAT, top_neighbor, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Exchange with bottom neighbor
    const float* my_bottom_rows_start = &local_data[(local_height - rows_to_exchange) * width * channels];
    MPI_Sendrecv(
        my_bottom_rows_start,                              // Send my bottom rows
        buffer_size, MPI_FLOAT, bottom_neighbor, 1,
        bottom_ghost_buffer.data(),                        // Receive into my bottom ghost buffer
        buffer_size, MPI_FLOAT, bottom_neighbor, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

std::vector<Keypoint> find_keypoints_and_descriptors(
    const Image& img, float sigma_min, int num_octaves, int scales_per_octave,
    float contrast_thresh, float edge_thresh, float lambda_ori,
    float lambda_desc) {
    assert(img.channels == 1 || img.channels == 3);
    const Image& input = (img.channels == 1) ? img : rgb_to_grayscale(img);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    std::chrono::high_resolution_clock::time_point start_time, end_time;
    if (rank == 0) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    ScaleSpacePyramid gaussian_pyramid_strip = generate_gaussian_pyramid_mpi(
        input, sigma_min, num_octaves, scales_per_octave);
    ScaleSpacePyramid dog_pyramid_strip =
        generate_dog_pyramid(gaussian_pyramid_strip);
    ScaleSpacePyramid grad_pyramid_strip =
        generate_gradient_pyramid(gaussian_pyramid_strip);
    vector<Keypoint> local_keypoints = find_keypoints(dog_pyramid_strip, contrast_thresh, edge_thresh);
   
    for(Keypoint &kp_tmp: local_keypoints) {
        
    }
    int initial_H = input.height * 2; // Full height of the base image in octave 0

for (Keypoint& kp : local_keypoints) {
    // 1. Determine the full height of the image for the keypoint's octave level.
    //    The height is halved at each successive octave.
    int octave_full_H = initial_H >> kp.octave;

    // 2. Calculate this process's starting row offset for that octave height.
    RowSplit split = split_rows(octave_full_H, world, rank);
    int y_offset = split.y0;

    // 3. Apply the offset to the keypoint's integer and sub-pixel coordinates.
    kp.j += y_offset; // This is the integer grid coordinate.
    kp.y += y_offset * MIN_PIX_DIST * std::pow(2, kp.octave); // This is the final sub-pixel coordinate.
}
    if (rank == 0) {
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed =
            end_time - start_time;
        cout << "1. generate gaussian dog gradient strip: " << elapsed.count()
             << " ms" << endl;
    }
    MPI_Datatype mpi_keypoint_type;
    create_mpi_keypoint_type(&mpi_keypoint_type);

    // 2. Each process sends the number of keypoints it found to the root
    int local_kp_count = local_keypoints.size();
    std::vector<int> all_kp_counts;
    if (rank == 0) {
        all_kp_counts.resize(world);
    }
    MPI_Gather(&local_kp_count, 1, MPI_INT,
               rank == 0 ? all_kp_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // 3. Root process prepares to receive all keypoints
    std::vector<Keypoint> tmp_kps;  // This will hold all keypoints on rank 0
    std::vector<int> displs;        // Displacements for Gatherv

    if (rank == 0) {
        displs.resize(world);
        displs[0] = 0;
        int total_kps = all_kp_counts[0];

        for (int i = 1; i < world; ++i) {
            total_kps += all_kp_counts[i];
            displs[i] = displs[i - 1] + all_kp_counts[i - 1];
        }
        tmp_kps.resize(total_kps);
    }

    // 4. Gather all keypoints from all processes into tmp_kps on the root
    MPI_Gatherv(local_keypoints.data(),                      // Send buffer
                local_kp_count,                              // Send count
                mpi_keypoint_type,                           // Send type
                rank == 0 ? tmp_kps.data() : nullptr,        // Recv buffer
                rank == 0 ? all_kp_counts.data() : nullptr,  // Recv counts array
                rank == 0 ? displs.data() : nullptr,         // Displacements array
                mpi_keypoint_type,                           // Recv type
                0,                                           // Root process
                MPI_COMM_WORLD);

    // 5. Don't forget to free the MPI type when you're done with it
    MPI_Type_free(&mpi_keypoint_type);

    ScaleSpacePyramid dog_pyramid;
    ScaleSpacePyramid gaussian_pyramid;
    ScaleSpacePyramid grad_pyramid;

    if (rank == 0) {
        start_time = std::chrono::high_resolution_clock::now();

        dog_pyramid.num_octaves = dog_pyramid_strip.num_octaves;
        dog_pyramid.imgs_per_octave = dog_pyramid_strip.imgs_per_octave;
        dog_pyramid.octaves.resize(dog_pyramid.num_octaves);

        gaussian_pyramid.num_octaves = gaussian_pyramid_strip.num_octaves;
        gaussian_pyramid.imgs_per_octave =
            gaussian_pyramid_strip.imgs_per_octave;
        gaussian_pyramid.octaves.resize(gaussian_pyramid.num_octaves);

        grad_pyramid.num_octaves = grad_pyramid_strip.num_octaves;
        grad_pyramid.imgs_per_octave = grad_pyramid_strip.imgs_per_octave;
        grad_pyramid.octaves.resize(grad_pyramid.num_octaves);
    }

    int H_o = input.height * 2;
    int W_o = input.width * 2;
    // std::vector<int> counts, displs;

    for (int oi = 0; oi < grad_pyramid_strip.num_octaves; ++oi) {
        if (rank == 0) {
            grad_pyramid.octaves[oi].resize(grad_pyramid_strip.octaves[oi].size());
        }

        // Calculate counts and displacements for a SINGLE channel plane.
        // This will be the same for both Gx and Gy gathers.
        std::vector<int> plane_counts(world), plane_displs(world);
        if (rank == 0) {
            for (int r = 0; r < world; ++r) {
                auto s = split_rows(H_o, world, r);
                plane_counts[r] = (s.y1 - s.y0) * W_o;  // Size of one channel's stripe
                plane_displs[r] = s.y0 * W_o;           // Displacement within one channel plane
            }
        }

        for (int si = 0; si < grad_pyramid_strip.octaves[oi].size(); ++si) {
            const Image& local_im = grad_pyramid_strip.octaves[oi][si];
            if (rank == 0) {
                // Allocate the final 2-channel image on the root
                grad_pyramid.octaves[oi][si] = Image(W_o, H_o, 2);
            }

            // --- Gather Channel 0 (Gx) ---
            const int plane_size = local_im.width * local_im.height;
            MPI_Gatherv(
                local_im.data,  // Sender points to start of Gx data
                plane_size, MPI_FLOAT,
                rank == 0 ? grad_pyramid.octaves[oi][si].data : nullptr,  // Receiver points to start of Gx plane
                rank == 0 ? plane_counts.data() : nullptr,
                rank == 0 ? plane_displs.data() : nullptr,
                MPI_FLOAT, 0, MPI_COMM_WORLD);

            // --- Gather Channel 1 (Gy) ---
            MPI_Gatherv(
                local_im.data + plane_size,  // Sender points to start of Gy data
                plane_size, MPI_FLOAT,
                rank == 0 ? grad_pyramid.octaves[oi][si].data + (W_o * H_o) : nullptr,  // Receiver points to start of Gy plane
                rank == 0 ? plane_counts.data() : nullptr,
                rank == 0 ? plane_displs.data() : nullptr,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        H_o /= 2;
        W_o /= 2;
    }

    if (rank == 0) {
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed =
            end_time - start_time;
        cout << "collect gauss and dog strips and : " << elapsed.count() << " ms"
             << endl;

        cout << "begin debug" << endl;
    }
    if (rank == 0) {
        // --- Timing for Find Keypoints ---
        // start_time = std::chrono::high_resolution_clock::now();
        // tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
        // end_time = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> elapsed_kps =
        //     end_time - start_time;
        // cout << "3. find_keypoints: " << elapsed_kps.count() << " ms" << endl;

        // --- Timing for Orientations and Descriptors ---
        start_time = std::chrono::high_resolution_clock::now();
        auto total_orientation_time =
            std::chrono::duration<double, std::milli>::zero();
        auto total_descriptor_time =
            std::chrono::duration<double, std::milli>::zero();

        std::vector<Keypoint> kps;
        cout << tmp_kps.size() << endl;
        kps.reserve(tmp_kps.size() * 2);
        if (rank == 0) {
            start_time = std::chrono::high_resolution_clock::now();
        }
#pragma omp parallel for
        for (Keypoint& kp_tmp : tmp_kps) {
            auto ori_start = std::chrono::high_resolution_clock::now();

            std::vector<float> orientations = find_keypoint_orientations(
                kp_tmp, grad_pyramid, lambda_ori, lambda_desc);
            auto ori_end = std::chrono::high_resolution_clock::now();
            total_orientation_time += (ori_end - ori_start);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                auto desc_start = std::chrono::high_resolution_clock::now();
                compute_keypoint_descriptor(kp, theta, grad_pyramid,
                                            lambda_desc);
                auto desc_end = std::chrono::high_resolution_clock::now();
                total_descriptor_time += (desc_end - desc_start);
                kps.push_back(kp);
            }
        }
        if (rank == 0) {
            end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_desc =
                end_time - start_time;
            cout << "computer oreintations adn decripstrso"
                 << elapsed_desc.count() << endl;
        }
        // cout << "5. compute_orientations_and_descriptors: " <<
        // elapsed_desc.count()
        //      << " ms" << endl;
        // cout << "   - 5a. find_keypoint_orientations (sum): "
        //      << total_orientation_time.count() << " ms" << endl;
        // cout << "   - 5b. compute_keypoint_descriptor (sum): "
        //      << total_descriptor_time.count() << " ms" << endl;

        return kps;
    }

    // Non-zero ranks return an empty vector as they did no work after the
    // pyramid generation.
    return {};
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b) {
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps) {
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}