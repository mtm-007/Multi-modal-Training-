import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

class MNISTDataset(Dataset):

    def __init__(self, data_df: pd.DataFrame, transform= None, is_test=False):
        super(MNISTDataset,self).__init__()
        dataset = []
        labels_positive = {}
        labels_negative = {}
        if is_test == False:
            # for each label create a set of same label images.
            for i in list(data_df.label.unique()):
                labels_positive[i] = data_df[data_df.label == i].to_numpy()
            # for each label create a set of image of different label.
            for i in list(data_df.label.unique()):
                labels_negative[i] = data_df[data_df.label != i].to_numpy()

        for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
            data = row.to_numpy()
            # if test then only image will be returned.
            if is_test:
                label = -1
                first = data.reshape(28, 28)
                second = -1
                dis = -1
            else:
                # label and image of the index for each row in df
                label = data[0]
                first = data[1:].reshape(28, 28)
                # probability of same label image == 0.5
                if np.random.randint(0, 2) == 0:
                    # randomly select same label image
                    second = labels_positive[label][
                        np.random.randint(0, len(labels_positive[label]))
                    ]
                else:
                    # randomly select different(negative) label 
                    second = labels_negative[label][
                        np.random.randint(0, len(labels_negative[label]))
                    ]
                # cosine is 1 for same and 0 for different label
                dis = 1.0 if second[0] == label else 0.0
                # reshape image
                second = second[1:].reshape(28, 28)

            # apply transform on both images
            if transform is not None:
                first = transform(first.astype(np.float32))
                if second is not -1:
                    second = transform(second.astype(np.float32))

            # append to dataset list. 
            # this random list is created once and used in every epoch
            # Convert to PyTorch tensors if not already
            if not isinstance(first, torch.Tensor):
                first = torch.from_numpy(first)
            first = first.to(torch.float32)
    
            if not is_test:
                if second is not -1:
                    if not isinstance(second, torch.Tensor):
                        second = torch.from_numpy(second)
                    second = second.to(torch.float32)
                else:
                    second = torch.tensor(-1, dtype=torch.float32)
                dis = torch.tensor(dis, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
    
            # append to dataset list
            if is_test:
                dataset.append((first, label))
            else:
                dataset.append((first, second, dis, label))
            
            #dataset.append((first, second, dis, label))

            
        
        self.dataset = dataset
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        # sample = self.dataset[i]  # Assuming this returns a tensor already
        # # Ensure dtype is float32
        # sample = sample.to(torch.float32)
        
        # return sample
        return self.dataset[i]
