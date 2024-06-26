import torch
from torch import nn, optim
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

#from src.mnist_dataset import MNISTDataset 
from Network import Network, ContrastiveLoss
from utils import show_images, load_latest_checkpoint
from mnist_dataset import MNISTDataset

from IPython.display import Image

import os
import warnings 
warnings.filterwarnings('ignore')

#device
device= "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device= "mps"


#load dataset 
data = pd.read_csv('../data/train.csv')
val_count =1000

#common transformation
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

#split the train to val and train
dataset = MNISTDataset(data.iloc[:-val_count], default_transform)
val_dataset = MNISTDataset(data.iloc[-val_count:], default_transform)

#setup Dataloaders with pytorch dataloaders
trainloader = DataLoader(
    dataset,
    batch_size = 16,
    shuffle =True,
    # pin_memory = True, # for faster data transfer speed btn CPU and GPU, but will consume more system memory
    # num_workers = 2,
    # prefetch_factor = 100,#to specify how many batches should be prefetched(loaded into memory[increased memory usage tho]) asynchronously in advance.
) 

#visualizing Datapoints

for batch_idx, (anchor_images, contrastive_images, distances, labels) in enumerate(trainloader):
    #converting tensors to numpy, numpy is easy to muniplate and display with matplotlib
    anchor_images = anchor_images.numpy()
    contrastive_images = contrastive_images.numpy()
    labels = labels.numpy()

    #display some imgages from batch
    show_images(anchor_images[:4], title = 'Anchor Images')
    show_images(contrastive_images[:4], title = '+/- Example Images')
    #break after displaying from one batch for demostration 
    break


net = Network()
net = net.to(device)
device

optimizer = torch.optim.AdamW(net.parameters(), lr = 3e-4)
loss_function = ContrastiveLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)


checkpoint_dir ='checkpoints/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def Train_model(epoch_count=100):
    net = Network()
    net = net.to(device)
    lrs = []
    losses = []

    for epoch in range(epoch_count):
        epoch_loss = 0
        batches = 0
        print('epoch-',epoch)
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        
        #lrs.append(optimizer.param_group[0]['lr'])
        print('learning rate',lrs[-1])

        for anchor, contrastive, distance, label in tqdm(trainloader):
            
            # Ensure data is in the correct shape
            assert anchor.shape[1] == 1, f"Expected anchor channels to be 1, but got {anchor.shape[1]}"
            assert contrastive.shape[1] == 1, f"Expected contrastive channels to be 1, but got {contrastive.shape[1]}"

            batches +=1
            optimizer.zero_grad()
            anchor_out = anchor.to(device, dtype=torch.float32)
            contrastive_out = contrastive.to(device, dtype=torch.float32)
            distance = distance.to(torch.float32).to(device)

            anchor_out = net(anchor_out)
            contrastive_out = net(contrastive_out)
            
            loss = loss_function(anchor_out, contrastive_out, distance)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        
        losses.append(epoch_loss.cpu().detach().numpy()/ batches)
        scheduler.step()
        print('epoch_loss', losses[-1])

        #save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch-{epoch}.pt')
        torch.save(net.state_dict(), checkpoint_path)
        
    return{
        "net": net,
        "losses":losses
    }

train = True # change False while not training
checkpoint_dir = 'checkpoints'

if train:
    training_result = Train_model()
    model = training_result["net"]
else:
    model = load_latest_checkpoint(checkpoint_dir)