# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:09:13 2021

@author: Joe
"""
import random

import torch
import torch.nn as nn

import numpy as np
import cv2

def conv_out_size(size, kernel_size=3, stride=2, padding=0):
    """
        Function to get the image size coming out of a convolutional layer
    """
    return ((np.array(size) - (kernel_size - 1) -1) // stride) + 1


class CustomCommonLayersCNN(nn.Module):
    def __init__(self,
                 num_numerical_inputs,
                 image_size,
                 num_outputs,
                 scale=1):
        super(CustomCommonLayersCNN, self).__init__()
                
        # kernel size 5, stride 3
        self.conv_0 = nn.Conv2d(
            1, int(scale*8), kernel_size=3, stride=2
        )
        # self.bn_0 = nn.BatchNorm2d(
        #     int(scale*8)
        # )
        self.conv_1 = nn.Conv2d(int(scale*8), int(scale*12), kernel_size=3, stride=2)
        # self.bn_1 = nn.BatchNorm2d(
        #     int(scale*12)
        # )
        
        # kernel size 3, stride 2
        self.conv_2 = nn.Conv2d(
            int(scale*12), int(scale*16), kernel_size=3, stride=2
        )
        # self.bn_2 = nn.BatchNorm2d(
        #     int(scale*16)
        # )
        
        # first 4 layers are common to all N CNN classifiers
        curr_img_size = conv_out_size(conv_out_size(conv_out_size(image_size[:2])), 5, 3)
        
        self.conv_3 = nn.Conv2d(
            int(scale*16), int(scale*18), kernel_size=5, stride=3
        )
        # self.bn_3 = nn.BatchNorm2d(
        #     int(scale*18)
        # )
        
        self.conv_4 = nn.Conv2d(
            int(scale*18), int(scale*20), kernel_size=3, stride=2
        )
        
        curr_img_size = conv_out_size(curr_img_size, 5, 3) # conv_out_size(conv_out_size(curr_img_size), stride=np.array([1,2]))
        num_data_points = num_numerical_inputs + \
            (curr_img_size[0] * curr_img_size[1] * int(scale*18))
            
        self.fully_connected_0 = nn.Linear(76, 30)
        self.fully_connected_1 = nn.Linear(30, num_outputs)
        
        self.prelu_fc_0 = nn.PReLU()
        self.prelu_fc_1 = nn.PReLU()
        #self.fully_connected_1 = nn.Linear(5, num_outputs)  
    
        
    def forward(
            self,
            numeric_input,
            input_image,
            activation_function,
            disp=False,
            disp_image_for_validation=False,
            return_extras=False
        ):
        # visit nn.functional, elu, relu, gelu, prelu(need a weight), softplus, tanh, sigmoid

        img_0 = torch.unsqueeze(input_image[:, 0, :, :], dim=1)
        img_1 = torch.unsqueeze(input_image[:, 1, :, :], dim=1)
        #out = None
        #print(input_image.shape, img_0.shape, img_1.shape)
        batch_size = img_0.shape[0]
        stacked = torch.cat([img_0, img_1], dim=0)
        #print(stacked.shape)
        # for img in [img_0, img_1]:
            #!print(img.shape)
        out_0 = activation_function(self.conv_0(
            stacked.type(torch.cuda.FloatTensor)
        ))
        #!print(out_0.shape)
        out_1 = activation_function(self.conv_1(out_0))
        #!print(out_1.shape)
        out_2 = activation_function(self.conv_2(out_1))
        #!print(out_2.shape)
        out_3 = activation_function(self.conv_3(out_2))
        #!print(out_3.shape)
        out_4 = activation_function(self.conv_4(out_3))
        #!print(out_4.shape)
        
            # out = out_4 if out is None else out_4 + out
        # # balance norm        
        # numeric_input[:, -3:-1] = numeric_input[:, -3:-1]/1000
        #!print(out.shape)
        #print(out_4.shape)
        #out_4_unstacked = torch.cat([out_4[:batch_size], out_4[batch_size:]], dim=1)
        #print( out_4[:batch_size].shape, out_4[batch_size:].shape)
        out_4_unstacked = out_4[:batch_size] + out_4[batch_size:]
        #print(out_4_unstacked.shape)
        
        union = torch.cat([
            torch.flatten(out_4_unstacked, 1, -1), 
            numeric_input,
        ], dim=1)
        
        #print(union.shape)
        

        out_5 = self.prelu_fc_0(self.fully_connected_0(union)) # activation_function
        #!print(out_5.shape)
        out_6 = self.prelu_fc_1(self.fully_connected_1(
            out_5
        )) # None
        #!print(out_6.shape)

        if disp:
            print('Starting Forward Pass..')
            for layer in [input_image, out_0, out_1, out_2, out_3, union, out_4, out_5]:
                print('\t Layer Size={}, min={}, max={}'.format(
                    layer.shape, layer.min(), layer.max()
                ))
                
        if disp_image_for_validation:
            img_test_0 = np.moveaxis(img_0[0].cpu().detach().numpy()*255, 0, 2).astype('uint8')
            img_test_1 = np.moveaxis(img_1[0].cpu().detach().numpy()*255, 0, 2).astype('uint8')
            
            cv2.imshow('img', np.concatenate([img_test_0, img_test_1], axis=0))
            cv2.waitKey(1000)
            cv2.destroyAllWindows() 
        
        #!print(out_6.shape, "\n\n\n")
        # if random.random() <= 0.0001:
        #      print(out_6[0])
        #print(self.fully_connected_0.weight.shape, torch.sum(self.fully_connected_0.weight, dim=0).shape)
        
        if return_extras:
            #print(torch.sum(self.fully_connected_0.weight, dim=0).shape, union.flatten().shape)
            return out_6, torch.sum(self.fully_connected_0.weight, dim=0), union.flatten()
        
        return out_6
 