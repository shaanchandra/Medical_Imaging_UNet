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
from torch.autograd import Variable
# import Progressbar
# progress = progressbar.ProgressBar()
from data import get_data, create_set, plot_sample_target
from model import UNet

import gc
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
class UNet_main():
    def __init__(self, img_path, target_idxs, in_h, in_b, out_h, out_b):
        super(UNet_main, self).__init__()
        self.train_imgs, self.train_labels, self.test_imgs, self.test_labels = get_data(img_path, target_idxs,in_h, in_b, out_h, out_b)
        self.input_height = in_h
        self.input_breadth = in_b
        self.output_height = out_h
        self.output_breadth = out_b
        
        
    def get_val_results(self, x_val, y_val, model):
        
        x_val = x_val.float()
        y_val = y_val.long()
        
        m = x_val.shape[0]
        with torch.no_grad():
            out = model(x_val)
        
        # out = out.permute(0, 2, 3, 1)
        out = out.resize(m*args.out_height*args.out_breadth, args.out_ch)
        labels = y_val.resize(m*width_out*height_out)
        loss = nn.CrossEntropyLoss(out, labels)
        
        correct = out.long().eq(labels.long()).cpu().sum().item()
        acc = correct/m
        
        return loss.item/m , acc
    
    def UNet_train(self):
                
        model = UNet(in_ch = args.in_ch, out_ch = args.out_ch, kernel_size = args.kernel_size)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.99)
        criterion = nn.CrossEntropyLoss()
        
        iters = np.ceil(self.train_imgs.size(0)/args.batch_size).astype(int)
        print("Steps per epoch = ", iters)
        best_acc = 0
        test_imgs = self.test_imgs
        test_labels = self.test_labels
        
        print("\n"+"="*40 + "\n\t Beginning Training \n" + "="*40)
        for epoch in range(args.epochs):
            train_loss = []
            
            # Shuffling the data
            permute_idxs = np.random.permutation(len(self.train_labels))
            train_imgs = self.train_imgs[permute_idxs]
            train_labels = self.train_labels[permute_idxs]
            
            for step in range(iters):
                start = step*args.batch_size
                stop = (step+1)*args.batch_size
                
                # Get batches
                train_batch_imgs = train_imgs[start:stop].float()
                train_batch_labels = train_labels[start: stop].long()
                
                # Get predictions
                optimizer.zero_grad()
                out = model(train_batch_imgs)
                
                # Calculate Loss
                # out = out.permute(0, 2, 3, 1)
                out = out.resize(args.batch_size*args.out_height*args.out_breadth, 2)
                # print(train_batch_labels.size())
                train_batch_labels = train_batch_labels.resize(args.batch_size*args.out_height*args.out_breadth)
                loss = criterion(out, train_batch_labels)
                
                # Backprop
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item)
                
            if epoch+1 % args.eval_every == 0:
                avg_train_loss = np.sum(train_loss)/iters
                correct = out.long().eq(train_batch_labels.long()).cpu().sum().item()
                train_acc = correct/iters
                
                avg_test_loss, test_acc = self.get_val_results(test_imgs, test_labels, model)
                
                if test_acc > best_acc:
                    best_acc = test_acc
            print("Epoch: {},  Train_loss = {:.4f},  Train_accuracy = {:.4f},  Valid_loss = {:.4f},   Val_accuracy = {:.4f}"
                  .format(epoch+1, avg_train_loss, train_acc, avg_test_loss, test_acc))
        print("\n"+"="*50 + "\n\t Training Done \n" + "="*50)
        print("\nBest Val accuracy = ", best_acc)


   
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="Path to save the model", type=str, default = './wic_results')
    parser.add_argument("--in_breadth", help = "height dimension of i/p image", type= int, default = 284)#496
    parser.add_argument("--in_height", help = "height dimension of i/p image", type= int, default = 284)#768
    parser.add_argument("--out_height", help = "height dimension of i/p image", type= int, default = 196)
    parser.add_argument("--out_breadth", help = "height dimension of i/p image", type= int, default = 196)
    parser.add_argument("--batch_size", help = "height dimension of i/p image", type= int, default = 9)
    parser.add_argument("--kernel_size", help = "size of convolution filter", type= int, default = 3)
    parser.add_argument("--in_ch", help = "No. of channels of input image (RGB = 3, Gray = 1)", type= int, default = 1)
    parser.add_argument("--out_ch", help = "No. of segmentation tags in labels", type= int, default = 2)
    parser.add_argument("--epochs", help = "height dimension of i/p image", type= int, default = 1000)
    parser.add_argument("--eval_every", help = "Evaluaiton frequency", type= int, default = 1)
    args = parser.parse_known_args()[0]
    
    in_h = args.in_height
    in_b = args.in_breadth
    out_h = args.out_height
    out_b = args.out_breadth
    
    print("Data directory has: ", os.listdir("../data"))

    #  Data from: https://www.kaggle.com/paultimothymooney/chiu-2015
    data_dir = os.path.join("..", "data", "2015_BOE_Chiu")
    img_path = [os.path.join(data_dir, 'Subject_0{}.mat'.format(i)) for i in range (1,10)] + [os.path.join(data_dir, 'Subject_10.mat')]
    
    # From the dataset description
    target_idxs = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
    
    plot_sample_target(img_path)
    unet = UNet_main(img_path, target_idxs, in_h, in_b, out_h, out_b)
    unet.UNet_train()


