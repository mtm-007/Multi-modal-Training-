import torch
from torch import nn,optim 
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  
from sklearn.decomposition import PCA
import umap
import umap.plot
import plotly.graph_objs as go 
import plotly.io as pio 
pio.renderers.default ='iframe'

import warnings 
warnings.filterwarnings('ignore')

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32, 5),
            nn.BatchNorm2d(32),
            #nn.GELU(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64, 5),
            nn.BatchNorm2d(64),
            #nn.GELU(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128, 5),
            nn.BatchNorm2d(128),
            #nn.GELU(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(0.3),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            #nn.GELU(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
        ) 
    def forward(self, x):
        x = self.conv1(x) # x: d * 32 * 12 * 12         
        x = self.conv2(x) # x: d * 64 * 4 * 4  
        #x = self.conv3(x)
        x = x.view(x.size(0), -1) # x: d * (64*4*4)
        x = self.linear1(x) # x: d *(64)       
        return x

#---------
class Network_wide(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(512, 64),
        )
    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = self.conv3(x)
        print("After conv3:", x.shape)
        x = self.conv4(x)
        print("After conv4:", x.shape)
        x = x.view(x.size(0), -1)
        print("After flatten:", x.shape)
        x = self.linear1(x)
        x = self.linear2(x) # x: d * 512
        x = self.linear3(x) # x: d * 64
        return x
       
    
    # def forward(self, x):
    #     x = self.conv1(x) # x: d * 32 * 12 * 12
    #     x = self.conv2(x) # x: d * 64 * 6 * 6
    #     x = self.conv3(x) # x: d * 128 * 3 * 3
    #     x = self.conv4(x) # x: d * 256 * 2 * 2
    #     x = x.view(x.size(0), -1) # x: d * (256*2*2)
    #     x = self.linear1(x) # x: d * 1024
    #     x = self.linear2(x) # x: d * 512
    #     x = self.linear3(x) # x: d * 64
    #     return x




class ContrastiveLoss_with_margin(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)
        self.margin = margin

    def forward(self, anchor, contrastive, distance):
        score = self.similarity(anchor, contrastive)
        similar_loss = (1 - distance) * torch.pow(1 - score, 2)
        dissimilar_loss = distance * torch.pow(torch.clamp(score - self.margin, min=0.0), 2)
        return torch.mean(similar_loss + dissimilar_loss)
#--------


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps = 1e-7)

    def forward(self, anchor, constrastive, distance):
        #distance is a tensor representing the target distance (0 for similar pairs, 1 for dissimilar pairs).
        score = self.similarity(anchor, constrastive)
        return nn.MSELoss()(score, distance) ##Ensures that the calculated score is close to the ideal distance (1 or 0)
    
class Network_t(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define the third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define the fully connected layer
        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        
        return x

    
