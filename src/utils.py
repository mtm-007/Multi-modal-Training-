import torch
from torch import nn,optim 
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  
from sklearn.decomposition import PCA
import math
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

def init_weights(model, gain=1.0):
    def init_module(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='gelu')
            nn.init.zeros_(m.bias)
    
    # Apply initialization to each module in the model
    model.apply(init_module)

def init_weights_for_gelu(m):
    if isinstance(m, nn.Conv2d):
        # Initialize Conv2d weights with a scaled normal distribution
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Initialize Linear weights with a scaled normal distribution
        in_features = m.weight.size(1)
        std = 1. / math.sqrt(in_features)
        m.weight.data.normal_(0, std)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)