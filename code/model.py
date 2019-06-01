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

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3):
        super(UNet, self).__init__()
        
        channels = [64,128,256,512]
        
        # Encoder block stacks
        self.encoder1 = self.encoder(in_ch= in_ch, out_ch= channels[0], kernel_size = 3)
        self.encoder2 = self.encoder(in_ch= channels[0], out_ch= channels[1], kernel_size = 3)
        self.encoder3 = self.encoder(in_ch= channels[1], out_ch= channels[2], kernel_size = 3)
        self.bridge = self.encoder(in_ch= channels[2], out_ch= channels[3], kernel_size = 3, final = True)
        
        # Max-pool layer
        self.max_pool = nn.MaxPool2d(kernel_size = 2)
        
        # De-conv layer
        self.deconv = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Decoder block stacks
        self.decoder3 = self.decoder(in_ch = channels[3], internal_ch=channels[2], out_ch=channels[1], kernel_size = 3)
        self.decoder2 = self.decoder(in_ch = channels[2], internal_ch=channels[1], out_ch=channels[0], kernel_size = 3)
        self.final = self.decoder(in_ch = channels[1], internal_ch=channels[0], out_ch= out_ch, kernel_size = 3, final = True)
        
    def forward(self, input):
        # Encoding sequence
        encoded1 = self.encoder1(input)
        pooled1 = self.max_pool(encoded1)
        encoded2 = self.encoder2(pooled1)
        pooled2 = self.max_pool(encoded2)
        encoded3 = self.encoder3(pooled2)
        pooled3 = self.max_pool(encoded3)
        encoded_final = self.bridge(pooled3)
        deconv = self.deconv(encoded_final)
        
        # Decoding Sequence
        decoder_ip_3 = self.crop_and_concat(deconv, encoded3, crop = True)
        decoded3 = self.decoder3(decoder_ip_3)
        decoder_ip_2 = self.crop_and_concat(decoded3, encoded2, crop = True)
        decoded2 = self.decoder2(decoder_ip_2)
        decoder_ip_1 = self.crop_and_concat(decoded2, encoded1, crop = True)
        output = self.final(decoder_ip_1)
        
        return output  
  
        
    def encoder(self, in_ch, out_ch, kernel_size = 3, final = False):
        relu = nn.ReLU()
        bn = nn.BatchNorm2d(out_ch)
        pool = nn.MaxPool2d(kernel_size = 2)
        deconv = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        if not final:
            encoder_orig = nn.Sequential(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size= kernel_size), relu, bn, 
                                      nn.Conv2d(in_channels = out_ch, out_channels = out_ch, kernel_size= kernel_size), relu, bn)
            # encoder_pooled = pool(encoder_orig)
            
        if final:
            encoder_orig = nn.Sequential(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size= kernel_size), relu, bn, 
                                      nn.Conv2d(in_channels = out_ch, out_channels = out_ch, kernel_size= kernel_size), relu, bn)
            # encoder_pooled = deconv(encoder_orig)
            
        return encoder_orig
    
    
    
    def decoder(self, in_ch, internal_ch, out_ch, kernel_size = 3, final =False):
        relu = nn.ReLU()
        
        if not final:
            decoder_block = nn.Sequential(nn.Conv2d(in_channels = in_ch, out_channels = internal_ch, kernel_size= kernel_size), relu, nn.BatchNorm2d(internal_ch), 
                                      nn.Conv2d(in_channels = internal_ch, out_channels = out_ch, kernel_size= kernel_size), relu, nn.BatchNorm2d(internal_ch),
                                      nn.ConvTranspose2d(in_channels=internal_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
        if final:
           decoder_block = nn.Sequential(nn.Conv2d(in_channels = in_ch, out_channels = internal_ch, kernel_size= kernel_size), relu, nn.BatchNorm2d(internal_ch), 
                                      nn.Conv2d(in_channels = internal_ch, out_channels = out_ch, kernel_size= kernel_size), relu, nn.BatchNorm2d(internal_ch),
                                      nn.Conv2d(in_channels = internal_ch, out_channels = out_ch, kernel_size= kernel_size), relu, nn.BatchNorm2d(out_ch)) 
        
        return decoder_block
    
    
    
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = torch.nn.functional.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)