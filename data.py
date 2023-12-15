import h5py, logging, random, sys, tqdm, os, re
from torch.utils.data import Dataset
import numpy as np 

#Training dataset class
class CBCTDataset(Dataset):
    def __init__(self, ifn, psz):
        self.psz = psz

        
        with h5py.File(ifn, 'r') as fp:
            self.features = fp['noise'][:].astype(np.float32)
            self.targets  = fp['ground_truth'][:].astype(np.float32)
        
        self.dim = self.features.shape
    
    def __getitem__(self, idx):
        if self.features.shape[-1] == self.psz:
            rst, cst = 0, 0
        else:
            rst = np.random.randint(0, self.features.shape[-2]-self.psz)
            cst = np.random.randint(0, self.features.shape[-1]-self.psz)

        inp = self.features[idx, np.newaxis, :, rst:rst+self.psz, cst:cst+self.psz]
        out = self.targets [idx, np.newaxis, :, rst:rst+self.psz, cst:cst+self.psz]
        
        return inp, out

    def __len__(self):
        return self.features.shape[0]

###################################################################################################

#utility functions for test data
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



class CBCT_Test_Dataset(Dataset):
    def __init__(self, ifn, dosage_level):

        #filter out test dir to clean volumes and specified dosage level
        ground_truth_paths = filter_and_sort_data(ifn, data_type='clean')
        input_paths = filter_and_sort_data(ifn, data_type=f'fdk_{dosage_level}')

        #initialize emtpy datasets
        self.ground_truth_data_set = np.zeros((len(ground_truth_paths), 256, 256, 256))
        self.input_data_set = np.zeros((len(input_paths), 256, 256, 256))

        #create the test datasets
        logging.info(f'\nCreating test data set ...')
        for i, (clean, ld) in enumerate(zip(ground_truth_paths, input_paths)):
            clean_volume = np.load(clean)
            input_volume = np.load(ld)

            self.ground_truth_data_set[i] = clean_volume
            self.input_data_set[i] = input_volume

        self.ground_truth_data_set = self.ground_truth_data_set.astype(np.float32)
        self.input_data_set = self.input_data_set.astype(np.float32)
        
        self.dim = self.input_data_set.shape
    
    def __getitem__(self, idx):

        inp = self.input_data_set[idx, np.newaxis]
        out = self.ground_truth_data_set [idx, np.newaxis]
        
        return inp, out

    def __len__(self):
        return self.input_data_set.shape[0]
