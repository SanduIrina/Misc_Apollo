#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from torch import *
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from cv_bridge import CvBridge
import os
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from rgb2yuv_yuv2rgb import *
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from cv2 import *

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net()

# outputs = net(images)

def callback(img):
    print("received on host")
    cvtColor(img, img, COLOR_YUV2RGB_UYVY)
    
    # img *= 1./255
    # cvtColor(img,img,1);
    # cv_img = CvBridge().imgmsg_to_cv2(img, desired_encoding="passthrough")
    # cv2.imwrite("file.png", cv_img)
    # n = torchvision.transforms.ToPILImage()(img)
    # imgTensor = torchvision.transforms.ToTensor()(n) 
    # outputs = net(imgTensor)
    # Send to AlexNet

def main():
    rospy.init_node('img_republisher')
    rospy.Subscriber("/steer_net", Image, callback, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    main()