import numpy as np
import torch
import torch.nn as nn

import sys
import os
import argparse
import datetime, time
current_date = datetime.datetime.now()

import warnings
warnings.simplefilter('ignore')

import scipy.io 
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from data import get_data, create_set, plot_sample_target
from model import UNet

import gc
gc.collect()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
    


def calc_elapsed_time(start, end):
        hours, rem = divmod(end-start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), seconds



class UNet_main():
    def __init__(self, img_path, checkpoint_path, target_idxs, in_h, in_b, out_h, out_b):
        super(UNet_main, self).__init__()
        self.train_imgs, self.train_labels, self.test_imgs, self.test_labels = get_data(img_path, target_idxs,in_h, in_b, out_h, out_b)
        self.input_height = in_h
        self.input_breadth = in_b
        self.output_height = out_h
        self.output_breadth = out_b
        self.model_path = checkpoint_path
        
        
    def get_val_results(self, x_val, y_val, model):
        
        x_val = x_val.float()
        y_val = y_val.long()
        loss_fn = nn.CrossEntropyLoss()
        
        m = x_val.shape[0]
        with torch.no_grad():
            out = model(x_val)
        
        # out = out.permute(0, 2, 3, 1)
        out = out.resize(m*args.out_height*args.out_breadth, args.out_ch)
        labels = y_val.resize(m*args.out_height*args.out_breadth)
        loss = loss_fn(out, labels)
        
        preds = torch.max(out.data,1)[1]
        correct = preds.long().eq(labels.long()).cpu().sum().item()
        acc = correct/(m*args.out_height*args.out_breadth)
        
        return loss.item() , acc


    def UNet_train(self):
                
        model = UNet(in_ch = args.in_ch, out_ch = args.out_ch, kernel_size = args.kernel_size).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.99)
        criterion = nn.CrossEntropyLoss()
        
        iters = np.ceil(self.train_imgs.size(0)/args.batch_size).astype(int)
        print("\nSteps per epoch =  {}\n".format(iters))
        best_acc = 0
        test_imgs = self.test_imgs
        test_labels = self.test_labels
        
        print("="*70 +"\n\t\t\t Training Network\n"+ "="*70)
        start = time.time()
        for epoch in range(args.epochs):
            print(epoch)
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
                # out = out.resize(args.batch_size * args.out_height * args.out_breadth, 2)
                # train_batch_labels = train_batch_labels.resize(args.batch_size * args.out_height * args.out_breadth)
                out = out.resize(train_batch_imgs.size(0)*args.out_height*args.out_breadth, args.out_ch)
                # print(train_batch_labels.size())
                train_batch_labels = train_batch_labels.resize(train_batch_labels.size(0)*args.out_height*args.out_breadth)
                loss = criterion(out, train_batch_labels)

                # Backprop
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                avg_train_loss = round(np.mean(train_loss),4)
                preds = torch.max(out.data,1)[1]
                correct = preds.long().eq(train_batch_labels.long()).cpu().sum().item()
                train_acc = correct/(iters*args.out_height*args.out_breadth)

                writer.add_scalar('Train/Loss', avg_train_loss, epoch+1)
                writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    writer.add_histogram('epochs/'+name, param.data.view(-1), global_step = epoch+1)
                
            if epoch % args.eval_every == 0:
                avg_test_loss, test_acc = self.get_val_results(test_imgs, test_labels, model)
                writer.add_scalar('Test/Loss', avg_test_loss, epoch+1)
                writer.add_scalar('Test/Accuracy', test_acc, epoch+1)
                if test_acc > best_acc:
                    best_acc = test_acc
                    print("\nNew High Score! Saving model...\n")
                    torch.save(model.state_dict(), self.model_path+"/model.pickle")

                end = time.time()
                h,m,s = calc_elapsed_time(start, end)
                print("\nEpoch: {}/{},  Train_loss = {:.4f},  Train_acc = {:.4f},   Val_loss = {:.4f},    Val_acc = {:.4f}"
                      .format(epoch+1, args.epochs, avg_train_loss, train_acc, avg_test_loss, test_acc))

        print("\n"+"="*50 + "\n\t Training Done \n")
        print("\nBest Val accuracy = ", best_acc)
        
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="Path to save the model", type=str, default = './checkpoints')
    parser.add_argument("--in_breadth", help = "height dimension of i/p image", type= int, default = 284)#496
    parser.add_argument("--in_height", help = "height dimension of i/p image", type= int, default = 284)#768
    parser.add_argument("--out_height", help = "height dimension of i/p image", type= int, default = 196)
    parser.add_argument("--out_breadth", help = "height dimension of i/p image", type= int, default = 196)
    parser.add_argument("--batch_size", help = "height dimension of i/p image", type= int, default = 2)
    parser.add_argument("--kernel_size", help = "size of convolution filter", type= int, default = 3)
    parser.add_argument("--in_ch", help = "No. of channels of input image (RGB = 3, Gray = 1)", type= int, default = 1)
    parser.add_argument("--out_ch", help = "No. of segmentation tags in labels", type= int, default = 2)
    parser.add_argument("--epochs", help = "height dimension of i/p image", type= int, default = 1000)
    parser.add_argument("--lr", help = "learning rate", type= float, default = 0.01)
    parser.add_argument("--eval_every", help = "Evaluation frequency", type= int, default = 1)
    args = parser.parse_known_args()[0]
    config = args.__dict__
    config['device'] = device

    #Print args
    print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
    for key, value in config.items():
        print(key + ' : ' + str(value))
    print("\n" + "x"*50)

    writer = SummaryWriter(os.path.join('logs', 'unet'))

    in_h = args.in_height
    in_b = args.in_breadth
    out_h = args.out_height
    out_b = args.out_breadth
    
    # print("Data directory has: ", os.listdir("../data"))

    #  Data from: https://www.kaggle.com/paultimothymooney/chiu-2015
    data_dir = os.path.join("..", "data")
    img_path = [os.path.join(data_dir, 'Subject_0{}.mat'.format(i)) for i in range (1,10)] + [os.path.join(data_dir, 'Subject_10.mat')]
    
    # From the dataset description
    target_idxs = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
    
    # Prepare checkpoint path for saving model
    checkpoint_path = os.path.join(args.save_path,"%02d_%02d_%02d__%02d_%02d" % (current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute))
    if not os.path.exists(checkpoint_path):
        print("\nCreating checkpoint directory... ", checkpoint_path)
        os.makedirs(checkpoint_path)

    # plot_sample_target(img_path)
    unet = UNet_main(img_path, checkpoint_path, target_idxs, in_h, in_b, out_h, out_b)
    unet.UNet_train()


