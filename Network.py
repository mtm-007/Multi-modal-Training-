import torch
from torch import nn,optim 
from torch.utils.data import DataLoader
from torchvision import transforms

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(0.3),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
        ) 
    def forward(self, x):
        x = self.conv1(x) # x: d * 32 * 12 * 12         
        x = self.conv2(x) # x: d * 64 * 4 * 4         
        x = x.view(x.size(0), -1) # x: d * (64*4*4)
        x = self.linear1(x) # x: d *(64)       
        return x


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps = 1e-7)

    def forward(self, anchor, constrastive, distance):
        #distance is a tensor representing the target distance (0 for similar pairs, 1 for dissimilar pairs).
        score = self.similarity(anchor, constrastive)
        return nn.MSELoss()(score, distance) ##Ensures that the calculated score is close to the ideal distance (1 or 0)