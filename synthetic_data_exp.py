from DatasetCompiler import DatasetCompiler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Module, Conv1d, Linear, ConvTranspose1d
import torch as T
import pandas as pd
import numpy as np
from torchsummary import summary

# TODO Put data into correct format for pytorch
# TODO Build the model

class VAEDataset(Dataset):

    def __init__(self, src):
        data_obj = DatasetCompiler.load_from_pickle(src)
        self.X = data_obj.x_train.astype('float32')
        self.y = data_obj.y_train.astype('float32')
        self.feature_names = data_obj.feature_names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index], self.y[index]]

    def get_splits(self, n_test=0.2):
        test_size = int(np.round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        return random_split(self, [train_size, test_size])

class VariationalAutoEncoder(Module):
    def __init__(self, input_size):

        # Define Encoder
        self.layer1 = Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=2)
        self.layer3 = Linear(in_features=100, out_features=2)

        # Define Decoder
        self.layer3 = Linear(in_features=7*7*64, out_features=2)
        self.layer4 = ConvTranspose1d(in_channels=64, )





    def sample(self, z_mean, z_sigma):
        # Pytroch randin doesnt let you state mean so we need to manually add it
        std = T.exp(0.5*z_sigma)
        epsilon = T.rand_like(std)
        sample = z_mean + (epsilon * std)
        return sample



# Prepare data for nerual network
dataset = VAEDataset('./data/processed/lrg_clean_data_v2.pickle')

# Train test split
train, test = dataset.get_splits()
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(train, batch_size=64, shuffle=False)

a= 0


