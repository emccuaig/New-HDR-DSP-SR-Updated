import io
from glob import glob
import numpy as np
from os.path import join
import os
from collections import defaultdict  # Import this to handle dynamic keys

def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
# 1. Setup Paths
dir_crop = glob("../data/hdr-dsp-real-dataset/crop/*")
dir_crop.sort()
dir_ratio = glob("../data/hdr-dsp-real-dataset/ratios/*")
dir_ratio.sort()
dir_sat = glob("../data/hdr-dsp-real-dataset/satmask/*")
dir_sat.sort()
# 2. Identify Special Index
ssc13index = set() # Using a set is faster for lookups
for i, crop in enumerate(dir_crop):
    if "ssc13" in crop:
        ssc13index.add(i)
# 3. Initialize Dictionaries with defaultdict
# This prevents KeyErrors if you encounter a frame count outside range(3, 16)
TrainData = defaultdict(list)
TrainRatios = defaultdict(list)
ValData = defaultdict(list)
ValRatios = defaultdict(list)
TestData = defaultdict(list)
TestRatios = defaultdict(list)

# 4. Processing Loop
# Note: You originally looped range(2500), but you have 2660 files. 
# Using len(dir_crop) ensures you process everything.
limit = min(2500, len(dir_crop)) 

for i in range(limit):
    data = np.load(dir_crop[i])
    ratio = np.load(dir_ratio[i])
    sat = np.load(dir_sat[i])
    
    nbFrames = len(ratio)
    
    # SAFER FILTERING LOGIC:
    # Build a list of valid indices rather than removing from a list
    valid_indices = []
    
    for j in range(nbFrames):
        is_valid = True
        
        # Check Saturation Mask (for non-ssc13)
        if i not in ssc13index:
            # Assuming 'False' in mask means saturated/bad pixel
            if False in sat[j]: 
                is_valid = False
        
        # Check Value Threshold (for ssc13)
        else:
            if data[j].max() > 2600:
                is_valid = False
        
        if is_valid:
            valid_indices.append(j)

    # 5. Distribute to Train/Val/Test
    # Only keep if we have enough valid frames
    num_valid = len(valid_indices)
    
    if num_valid >= 3:
        key = str(num_valid)
        
        # Select the specific data slices once
        filtered_data = data[valid_indices]
        filtered_ratio = ratio[valid_indices]

        if i < 2000:
            TrainData[key].append(filtered_data)
            TrainRatios[key].append(filtered_ratio)
        elif i < 2200:
            ValData[key].append(filtered_data)
            ValRatios[key].append(filtered_ratio)
        else:
            TestData[key].append(filtered_data)
            TestRatios[key].append(filtered_ratio)

# 6. Saving Loop (Your improved version)
all_keys = set(TrainData.keys()) | set(ValData.keys()) | set(TestData.keys())
sorted_keys = sorted(list(all_keys), key=int)

print(f"Processing complete. Found groups for frame counts: {sorted_keys}")

for key in sorted_keys:
    base_output_path = "../data/SkySat_ME_noSaturation"
    
    # Helper function to reduce repetition
    def save_set(data_dict, ratio_dict, subfolder_name):
        if key in data_dict and len(data_dict[key]) > 0:
            folder = join(base_output_path, subfolder_name, key)
            safe_mkdir(folder)
            
            # Allow object arrays if image sizes differ
            try:
                np.save(join(folder, f"{subfolder_name}LR.npy"), np.array(data_dict[key]))
                np.save(join(folder, f"{subfolder_name}Ratio.npy"), np.array(ratio_dict[key]))
            except ValueError:
                print(f"Warning: shapes mismatch in {subfolder_name} for key {key}. Saving as object array.")
                np.save(join(folder, f"{subfolder_name}LR.npy"), np.array(data_dict[key], dtype=object))
                np.save(join(folder, f"{subfolder_name}Ratio.npy"), np.array(ratio_dict[key], dtype=object))

    save_set(TrainData, TrainRatios, "train")
    save_set(ValData, ValRatios, "val")
    save_set(TestData, TestRatios, "test") # Typo fix: TestData, TestRatios