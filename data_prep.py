

import glob, os, scipy.io, matplotlib, re, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image as im
from tqdm import tqdm

#path to data 
data_path = '...'

##################################
#utility functions
def natural_sorted(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", str(x))]
    return sorted(l, key=key)

def filter_and_sort_data(ddir, data_type):
    res = []
    for dirpath, dnames, fnames in os.walk(f"{ddir}/"):
        for f in fnames:
            if data_type in f:
                res.append(os.path.join(dirpath, f))
                
    return natural_sorted(res)

##########################################


#Collect Clean FDK volumes (Ground Truth)
clean_paths = filter_and_sort_data(data_path, data_type='clean')


#Collect Clinical Dose Volumes (Input 1)
#clinical/low dose
#fdk_cd_paths = filter_and_sort_data(data_path, data_type='fdk_clinical_dose')
fdk_cd_paths = filter_and_sort_data(data_path, data_type='fdk_low_dose')

#initialize empty dataset of size (#samples, 256, 256, 256)
ground_truth_data_set = np.zeros((len(clean_paths), 256, 256, 256))
input_data_set = np.zeros((len(fdk_cd_paths), 256, 256, 256))

for i, (clean, cd) in tqdm(enumerate(zip(clean_paths, fdk_cd_paths))):
    clean_volume = np.load(clean)
    input_volume = np.load(cd)

    ground_truth_data_set[i] = clean_volume
    input_data_set[i] = input_volume


#sanity check
plt.figure(figsize=(10,10))
plt.imshow(ground_truth_data_set[5][10], cmap='gray')
plt.savefig('ground_truth_example.png')
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(input_data_set[5][10], cmap='gray')
plt.savefig('input_example.png')
plt.close()


out_path = '...'

with h5py.File(out_path, 'w') as fp:
    fp.create_dataset('noise', data=input_data_set)
    fp.create_dataset('ground_truth', data=ground_truth_data_set)
