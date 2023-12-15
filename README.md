# ICASSP-2024 3D-CBCT Grand Challenge: Advancing the frontiers of deep learning for low-dose 3D cone-beam CT reconstruction   

## [Grand Challenge Website](https://sites.google.com/view/icassp2024-spgc-3dcbct/home)

## Overview   

The challenge seeks to push the limits of deep learning algorithms for 3D cone beam computed tomography (CBCT) reconstruction from low-dose projection data (sinogram). The key objective in medical CT imaging is to reduce the X-ray dose while maintaining image fidelity for accurate and reliable clinical diagnosis. In recent years, deep learning has been shown to be a powerful tool for performing tomographic image reconstruction, leading to images of higher quality than those obtained using the classical solely model-based variational approaches. Notwithstanding their impressive empirical success, the best-performing deep learning methods for CT (e.g., algorithm unrolling techniques such as learned primal-dual) are not scalable to real-world CBCT clinical data. Moreover, the academic literature on deep learning for CT generally reports the image recovery performance on the 2D reconstruction problem (on a slice-by-slice basis) as a proof-of-concept. Therefore, in order to have a fair assessment of the applicability of these methods for real-world 3D clinical CBCT, it is imperative to set a benchmark on an appropriately curated medical dataset. <strong>The main goal of the challenge is to encourage deep learning practitioners and clinical experts to develop novel deep learning methodologies (or test existing ones) for clinical low-dose 3D CBCT imaging with different dose levels</strong>.    

## Contribution    

Our primary focus is on denoising. Therefore, we develop a deep learning approach that takes the clinical/low dose volume and outputs the corresponding clean volume.   

## Methodology   

Essentially, denoising can be framed as an image translation task in which one wishes to translate the noisy image into its denoised counterpart. This type of problem has been well studied in the literature with numerous approaches developed. However, this challenge focuses on data samples consisting of 3D volumes. Therefore, one needs to extend the 2D approaches into 3D, usually only requiring changing 2D convolutional layers to 3D. Our solution presented here consists of a simple [3D UNet](https://arxiv.org/pdf/1606.06650.pdf) architecture. While we examined newer solutions based on [UNet++](https://arxiv.org/pdf/1807.10165.pdf) and [Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf), we found the simple UNet to be superior showing that nature favors simplicity.    

## Installation   

### On Linux   

Create a conda environment with:   
```bash
conda env create -f environment.yml
conda activate GrandChallenge
```

## Running the Project   

### Data   

The data can be accessed [here](https://sites.google.com/view/icassp2024-spgc-3dcbct/data). The dataset consists of 1010 data points with shape (256, 256, 256). The data is divided into 800 training samples, 100 validation samples, and 110 test samples. We provide the training and validation datasets only, with the ground truth data, two sinograms, and two noisy reconstructions using the FDK algorithm (the cone-beam equivalent of FBP) corresponding to the two different levels of noise as stated above. The format is ```numpy``` arrays with the following naming convention:

{patient ID}_clean_fdk_256.npy &rarr; Ground truth/target    
{patient ID}_fdk_clinical_dose_256.npy &rarr; FDK for clinical dose    
{patient ID}_fdk_low_dose_256.npy &rarr; FDK for low dose    
{patient ID}_sino_clinical_dose.npy &rarr; Sinogram for clinical dose    
{patient ID}_sino_low_dose.npy &rarr; Sinogram for low dose    

The training data is 217 Gb once extracted and the validation data is 37 Gb once extracted.   

### Training   

#### Data Format   

The original format of the data is a ```numpy``` array for each volume. To make loading the data for training/validation easier, we first collect all the training volumes into a single ```numpy``` array and save it as an ```h5``` file using ```h5py```. Therefore, the training file consists of the paired input/target volumes for training/validation. This is done using the ```data_prep.py``` python script that requires specifying:   

-path to training/validation volumes     
-data_type parameter used in ```filter_and_sort_data``` function: fdk_low_dose/fdk_clinical_dose (filter out the specified dosage level)     
-out path: path to save the h5 data    

#### Model Training    

The code used to train the model is in ```main.py```. Due to the size of the data, we use the ```DistributedDataParallel``` framework in ```PyTorch``` to train on multiple GPUs within a single node. After activating the conda environment, the bare minimum to run the code is:     

```bash
python -m torch.distributed.run --nproc_per_node=... main.py
```
where ```...``` requires specifying how many GPUs are available.   

However, there are a few command line arguments that can be specified that may be helpful:   

-mbsz(int): minibatch size (default 2)    
-psz(int): volume patch size (default 256)    
-aug(int): data augmentation (default 0/False)    
-maxep(int): number of training epochs (default 100)     
-lr(float): learning rate (default 1e-4)    
-mdl(str): path to model for continued training of a save model (default None)        

During training, the best model based on the validation loss is saved. Furthermore, the training and validation losses are saved every epoch and random 2D slice from a validation volume is saved every 10th epoch so that the model progressed can be viewed over time.    

#### Inference   

The code used to test the model is in ```infer.py```. To run the script you will need to specify the location of the data with the ```test_dir``` variable. Since we trained a separate model for the clinical and low dose dosage level, you will need to specify the dosage level using the ```dosage_level``` variable. To conform with the challenge, the data is filtered and paired on the fly. This is hidden in the ```data.py``` file. After you can run inference using: 

```bash
python infer.py
```

The script prints the mse results to the terminal and saves the results in a ```.csv``` file.   
