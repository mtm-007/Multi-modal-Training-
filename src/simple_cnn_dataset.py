import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch



class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx] # introduced for pandas dataframe compatiblity
        #image = row['pixel0':'pixel783'].values  # Adjust column names as needed
        image = row['pixel0':'pixel783'].values.reshape(28, 28)  # Assuming 28x28 images
        
        #print("This is to check before applying",image.max())
        # Convert to uint8
        image = (image / image.max() * 255).astype(np.uint8)
    
        label = row['label']  # Adjust column name as needed
        if self.transform:
            image = self.transform(image)
        return image, label
        

   