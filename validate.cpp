#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>

struct KeypointData {
    int i, j;                    // discrete coordinates
    int octave, scale;           // octave and scale
    std::vector<int> descriptor; // 128-dimensional descriptor
    
    bool operator<(const KeypointData& other) const {
        if (i != other.i) return i < other.i;
        

        if (j != other.j) return j < other.j;
        
        if (octave != other.octave) return octave < other.octave;
        
        if (scale != other.scale) return scale < other.scale;
        
        for (size_t i = 0; i < descriptor.size() && i < other.descriptor.size(); i++) {
            if (descriptor[i] != other.descriptor[i]) {
                return descriptor[i] < other.descriptor[i];
            }
        }
        
        return descriptor.size() < other.descriptor.size();
    }
};

std::vector<KeypointData> read_keypoints(const std::string& filename) {
    std::vector<KeypointData> keypoints;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return keypoints;
    }
    
    int num_keypoints;
    file >> num_keypoints;
    std::cout << "Reading " << num_keypoints << " keypoints from " << filename << std::endl;
    
    keypoints.resize(num_keypoints);
    
    for (int k = 0; k < num_keypoints; k++) {
        KeypointData& kp = keypoints[k];
        
        file >> kp.i >> kp.j >> kp.octave >> kp.scale;
        
        kp.descriptor.resize(128);
        for (int i = 0; i < 128; i++) {
            file >> kp.descriptor[i];
        }
    }
    
    file.close();
    return keypoints;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <output.txt>" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string gold_file = argv[2];
    
    std::vector<KeypointData> input_keypoints = read_keypoints(input_file);
    std::vector<KeypointData> gold_keypoints = read_keypoints(gold_file);
    
    if (input_keypoints.empty() || gold_keypoints.empty()) {
        std::cerr << "Failed to read keypoints" << std::endl;
        return 1;
    }
    
    std::sort(input_keypoints.begin(), input_keypoints.end());


     auto descriptor_distance = [](const std::vector<int>& desc1, const std::vector<int>& desc2) -> double {
         double sum = 0.0;
         for (int i = 0; i < 128; ++i) {
             int diff = (int)desc1[i] - (int)desc2[i];
             sum += diff * diff;
         }
         return std::sqrt(sum);
     };


     size_t n = gold_keypoints.size();
     std::cout << "Analyzing " << n << " golden keypoints" << std::endl;

     size_t total_matches = 0;

     for (size_t idx = 0; idx < n; ++idx) {
         const auto& golden_kp = gold_keypoints[idx];

         std::vector<KeypointData> same_group;
         
         size_t left = 0, right = input_keypoints.size();
         size_t first_match = input_keypoints.size();
         
         while (left < right) {
             size_t mid = left + (right - left) / 2;
             const auto& input_kp = input_keypoints[mid];
             
             if (input_kp.i < golden_kp.i || 
                 (input_kp.i == golden_kp.i && input_kp.j < golden_kp.j)) {
                 left = mid + 1;
             } else {
                 first_match = mid;
                 right = mid;
             }
         }
         
         for (size_t i = first_match; i < input_keypoints.size(); ++i) {
             const auto& input_kp = input_keypoints[i];
             
             if (input_kp.i == golden_kp.i && input_kp.j == golden_kp.j) {
                 same_group.push_back(input_kp);
             } else {
                 break;
             }
         }

         int group_size = same_group.size();


         if (group_size > 0) {
             double min_dist = 100000000.0;
             for (const auto& kp : same_group) {
                 double dist = descriptor_distance(kp.descriptor, golden_kp.descriptor);
                 if (dist < min_dist) {
                     min_dist = dist;
                 }
             }

             if (min_dist < 1.5) {
                 total_matches++;
             }
            //  std::cout << "  Min distance: " << std::fixed << std::setprecision(5) << min_dist << std::endl;
         }
     }

     std::cout << "match percentage: " << (static_cast<double>(total_matches) / n) << std::endl;

    
    return 0;
}
