#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <mpi.h>

#include "image.hpp"
#include "sift.hpp"


#include <mpi.h>
// ...

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // args check only once is fine; all ranks see same argc
    if (argc != 4) {
        if (rank == 0)
            std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];

    auto start_overall = std::chrono::high_resolution_clock::now();

    // All ranks can load the input (simple + avoids extra broadcast)
    Image img(input_img);
    img = (img.channels == 1) ? img : rgb_to_grayscale(img);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> kps = find_keypoints_and_descriptors(img);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> dt = t1 - t0;
        std::cout << "find_keypoints_and_descriptors took: " << dt.count() << " ms\n";
    }

    // =========================
    // ROOT-ONLY OUTPUT SECTION
    // =========================
    if (rank == 0) {
        // write txt
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale;
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
        }

        // write image with keypoints
        Image result = draw_keypoints(img, kps);
        result.save(output_img);
    }

    // Optional: make sure root finishes writing before exit
    MPI_Barrier(MPI_COMM_WORLD);

    auto end_overall = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> dt = end_overall - start_overall;
        std::cout << "Execution time: " << dt.count() << " ms\n";
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }

    MPI_Finalize();
    return 0;
}
