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

def show_images(images, title =''):
    num_images = len(images)
    fig,axes = plt.subplots(1, num_images,figsize=(9,3))
    for i in range(num_images):
        img = np.squeeze(images[i])
        axes[i].imshow(img,cmap='gray')
        axes[i].axis('off')
    fig.suptitle(title)
    plt.show()

def load_latest_checkpoint(checkpoint_dir='checkpoints'):
    # List all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch') and f.endswith('.pt')]
    
    # Extract epoch numbers
    epochs = [int(re.findall(r'\d+', f)[0]) for f in checkpoint_files]
    
    # Find the latest epoch
    latest_epoch = max(epochs)
    latest_checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{latest_epoch}.pt')
    
    # Load the latest checkpoint
    checkpoint = torch.load(latest_checkpoint_path)
    net = Network()
    net.load_state_dict(checkpoint)
    net.eval()
    
    return net