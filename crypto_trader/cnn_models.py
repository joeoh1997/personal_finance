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
            
        self.fully_connected_0 = nn.Linear(76, 200)
        self.fully_connected_1 = nn.Linear(200, num_outputs)
        
        self.prelu_fc_0 = nn.PReLU()
        self.prelu_fc_1 = nn.PReLU() 
    
        
    def forward(
            self,
            numeric_input,
            input_image,
            activation_function,
            disp=False,
            disp_image_for_validation=False,
            return_extras=False,
            include_5_min=False
        ):

        imgs = torch.unsqueeze(input_image[:, 0, :, :], dim=1)
        imgs_5_min = torch.unsqueeze(input_image[:, 1, :, :], dim=1)

        batch_size = img_0.shape[0]

        if include_5_min:
            imgs = torch.cat([imgs, imgs_5_min], dim=0)

        out_0 = activation_function(self.conv_0(
            imgs.type(torch.cuda.FloatTensor)
        ))

        out_1 = activation_function(self.conv_1(out_0))
        out_2 = activation_function(self.conv_2(out_1))
        out_3 = activation_function(self.conv_3(out_2))
        out_4 = activation_function(self.conv_4(out_3))

        if include_5_min:
            out_4 = out_4[:batch_size] + out_4[batch_size:]
        
        union = torch.cat([
            torch.flatten(out_4, 1, -1), 
            numeric_input,
        ], dim=1)

        out_5 = self.prelu_fc_0(self.fully_connected_0(union))
        out_6 = self.prelu_fc_1(self.fully_connected_1(
            out_5
        ))

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
        
        if return_extras:
            return out_6, torch.sum(self.fully_connected_0.weight, dim=0), union.flatten()
        
        return out_6
 