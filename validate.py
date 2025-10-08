from skimage.metrics import structural_similarity as ssim
import cv2
import subprocess
import sys
import re
import argparse
import os

def check_descriptor(input_file, golden_file):
    try:
        result = subprocess.run(['./validate', input_file, golden_file], 
                              capture_output=True, text=True, check=True)
        output = result.stdout
        
        match_percentage = None
        for line in output.split('\n'):
            if 'match percentage:' in line:
                match = re.search(r'match percentage:\s*([\d.]+)', line)
                if match:
                    match_percentage = float(match.group(1))
                break
        
        return match_percentage, output
    except subprocess.CalledProcessError as e:
        print(f"Error running validate: {e}")
        return None, e.stderr
    except FileNotFoundError:
        print("Error: validate executable not found")
        return None, "validate executable not found"


if len(sys.argv) != 5:
    print("Usage: python3 validate.py ./results/xx.txt ./goldens/xx.txt ./results/xx.jpg ./goldens/xx.jpg")
    sys.exit(1)

test_txt = sys.argv[1]
golden_txt = sys.argv[2]
test_image = sys.argv[3]
golden_image = sys.argv[4]


if not os.path.exists(test_txt):
    print(f"Error: {test_txt} not found")
    sys.exit(1)
if not os.path.exists(golden_txt):
    print(f"Error: {golden_txt} not found")
    sys.exit(1)
if not os.path.exists(test_image):
    print(f"Error: {test_image} not found")
    sys.exit(1)
if not os.path.exists(golden_image):
    print(f"Error: {golden_image} not found")
    sys.exit(1)

img1 = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(golden_image, cv2.IMREAD_GRAYSCALE)

descriptor_match_percentage, validate_output = check_descriptor(test_txt, golden_txt)

if descriptor_match_percentage is not None and validate_output is not None:
    # print(f"Descriptor match percentage: {descriptor_match_percentage:.6f}")
    score, diff = ssim(img1, img2, full=True)
    # print(f"SSIM: {score:.6f}")

    if descriptor_match_percentage >= 0.98 and score >= 0.98:
        print("Pass")
    else:
        print("Wrong")
else:
    print("Failed to get descriptor match percentage")
    print("Validate output:", validate_output)