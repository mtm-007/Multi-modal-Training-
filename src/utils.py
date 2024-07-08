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


import torch.nn.init as init
import wandb
import random
import math,os,sys,io

from collections import defaultdict
from tqdm import tqdm
from PIL import Image  # Add this import statement

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

def plot_activation_stats(activations_list):
    if not activations_list:
        print("No activation data to plot.")
        wandb.log({'message':'No activation data to plot'})
        return

    for layer_name in activations_list[0].keys():
        means = [epoch[layer_name]['mean'] for epoch in activations_list]
        vars = [epoch[layer_name]['var'] for epoch in activations_list]
        neg_ratios = [epoch[layer_name]['neg_ratio'] for epoch in activations_list]

        # Ensure data is flattened and converted to numpy arrays
        means = np.array(means).flatten()
        vars = np.array(vars).flatten()
        neg_ratios = np.array(neg_ratios).flatten()

        # logging this stat only on wandb 
        #print(f'Plotting data for layer: {layer_name}')
        wandb.log({'Plotting data for layer':layer_name}) # logging this stat only on wandb 
        #print(f'Means: {means}')
        wandb.log({'Means': means})
        #print(f'Variances: {vars}')
        wandb.log({'Variances': vars})
        #print(f'Negative Ratios: {neg_ratios}')
        wandb.log({'Negative Ratios':neg_ratios})

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(means)
        plt.title(f'{layer_name} - Mean Activation')
        plt.xlabel('Batch')
        plt.ylabel('Mean')
        
        plt.subplot(132)
        plt.plot(vars)
        plt.title(f'{layer_name} - Activation Variance')
        plt.xlabel('Batch')
        plt.ylabel('Variance')

        plt.subplot(133)
        plt.plot(neg_ratios)
        plt.title(f'{layer_name} - Negative Activation Ratio')
        plt.xlabel('Batch')
        plt.ylabel('Ratio')
         
        plt.tight_layout()
        

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert BytesIO to PIL Image
        image = Image.open(buf)

        # Log the plot to Weights and Biases
        wandb.log({f'{layer_name} activations': wandb.Image(image)})
        plt.show() # this should be after wandb log 

        plt.close()  # Close the figure to free up memory
        buf.close()

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


def initialize_weights_mod(model, activation_function='relu'):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if activation_function == 'relu':
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif activation_function == 'leaky_relu':
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)


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