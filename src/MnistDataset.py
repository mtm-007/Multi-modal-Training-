import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):

    def __init__(self, data_df: pd.DataFrame, transform=None, is_test=False):
        super(MNISTDataset, self).__init__()
        self.transform = transform
        self.is_test = is_test
        
        dataset = []
        labels_positive = {}
        labels_negative = {}
        
        if not is_test:
            # Create dictionaries of positive and negative samples for each label
            for label in data_df['label'].unique():
                labels_positive[label] = data_df[data_df['label'] == label].to_numpy()
                labels_negative[label] = data_df[data_df['label'] != label].to_numpy()

        for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
            data = row.to_numpy()
            
            if is_test:
                # Test case: return only the image (first)
                first = data[1:].reshape(28, 28)
                second = None
                dis = -1
                label = -1
            else:
                # Training case: return both images, dissimilarity flag, and label
                label = data[0]
                first = data[1:].reshape(28, 28)
                
                if np.random.randint(0, 2) == 0:
                    second = labels_positive[label][np.random.randint(0, len(labels_positive[label]))][1:].reshape(28, 28)
                else:
                    second = labels_negative[label][np.random.randint(0, len(labels_negative[label]))][1:].reshape(28, 28)
                
                dis = 1.0 if second is not None and second[0] == label else 0.0
            
            # Apply transformations if provided
            if self.transform:
                first = self.transform(first.astype(np.float32))
                if second is not None:
                    second = self.transform(second.astype(np.float32))
            
            dataset.append((first, second, dis, label))
        
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        first, second, dis, label = self.dataset[i]

        # Convert images to float32 tensors if not None
        if first is not None:
            first = first.to(torch.float32)
        if second is not None:
            second = second.to(torch.float32)

        return first, second, dis, label
