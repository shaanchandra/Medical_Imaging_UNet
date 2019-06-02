import numpy as np
import torch
import torch.nn as nn

import sys
import os
import argparse

import scipy.io 
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
# import Progressbar
# progress = progressbar.ProgressBar()

def create_set(path_list, target_idxs, in_h, in_b, out_h, out_b):
    imgs = []
    labels = []
    
    for path in tqdm(path_list):
        mat = scipy.io.loadmat(path)
        x = mat['images']
        y = mat['manualFluid1']
        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        
        # Batch first
        x = x.transpose(2,0,1)/255
        x = resize(x, (x.shape[0],in_b, in_h))
        x = torch.from_numpy(x)
        
        y = y.transpose(2,0,1)
        y = resize(y, (y.shape[0],out_b, out_h))
        y = torch.from_numpy(y)
        y = torch.where(y==0, torch.zeros(y.shape), torch.ones(y.shape))
        
        for idx in target_idxs:
            imgs += [torch.unsqueeze(x[idx],0)]
            labels += [torch.unsqueeze(y[idx],0)]
            
    return torch.stack(imgs), torch.stack(labels)
        
        

def get_data(img_path, target_idxs,in_h, in_b, out_h, out_b):
    
    print("="*40 +"\n\t Getting Training data\n"+ "="*40)
    train_imgs, train_labels = create_set(img_path[:9], target_idxs, in_h, in_b, out_h, out_b)
    print("="*40 +"\n\t Getting Test data\n"+ "="*40)
    test_imgs, test_labels = create_set(img_path[9:], target_idxs, in_h, in_b, out_h, out_b)
    print("\n"+"-"*60)
    print("No. of training examples: ", train_imgs.size(0))
    print("No. of test examples: ", test_imgs.size(0))
    
    return train_imgs, train_labels, test_imgs, test_labels


def plot_sample_target(img_path):   
    print("\nPlotting sample image and target...\n")
    mat = scipy.io.loadmat(img_path[0])
    img_tensor = mat['images']
    manual_fluid_tensor_1 = mat['manualFluid1']
    
    img_tensor = torch.from_numpy(img_tensor)
    manual_fluid_tensor_1 = torch.from_numpy(manual_fluid_tensor_1)
    
    img_array = img_tensor.permute(2, 0, 1)
    manual_fluid_array = manual_fluid_tensor_1.permute(2, 0, 1)
    plt.figure(1)
    plt.imshow(img_array[15])
    plt.figure(2)
    plt.imshow(manual_fluid_array[15])     
        