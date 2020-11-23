import pandas as pd
import os
import importlib
from src.utils.remove import remove
from src.utils.set_path import path
from sklearn.model_selection import train_test_split
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.ImageSaver import ImageSaver
import wandb
from tqdm import tqdm
from warnings import warn


"""
Dataset class acts as a single source of truth for all data and is consumed by Model Objects.

Expected Format for Datasets
    Each datasets are expected to csvs
    Datasets will be names - dataset Name/dataset component name 
    Datasets can be organised as follows:
            - root
                - Dataset name 1
                        - Dataset components (csv files only)
                - Dataset name 2
                        - Dataset components
    
Constructor Arguments
    src - List of file paths to dataset directories
    transforms = List of functions used to transform datasets
"""


class DatasetCompiler():

    def __init__(self, src, y_labels, test_size):


        # User defined attributes
        self.src = path(src)
        self.y_labels = y_labels
        self.test_size = test_size

        # default attributes
        self.datasets = {}
        self.applied_transforms = []


    def load(self):

        folder_list = os.listdir(self.src)
        # Handle .DS_store for mac
        folder_list = remove(folder_list, '.DS_Store')

        for folder in folder_list:
            file_list = os.listdir(os.path.join(self.src, folder))
            file_list = [file for file in file_list if file.endswith('.csv')]

            file_list = remove(file_list, '.DS_Store')
            for file in file_list:
                data = pd.read_csv(os.path.join(self.src, folder, file))
                file_name = file.split('.')
                dataset_name = f'{folder}-{file_name[0]}'

                self.datasets[dataset_name] = {'data': data}

    def add_feature(self, feature_name: str, feature: pd.Series, names=None):
        """
        :param feature_name: name of the column to add to pandas df
        :param feature: a pandas series of values to add as a column
        :param names: names of the datasets to apply this feature onto
        """
        if names is None or len(names) == 0:
            # add to all datasets
            for name in self.datasets.keys():
                if self.len(name) != len(feature):
                    raise ValueError(f"Length ({self.len(name)}) of dataset {name} is not equal to length ({len(feature)}) of series {feature_name}")
                self.datasets[name]["data"][feature_name] = feature

        else:
            for name in names:
                if self.len(name) != len(feature):
                    raise ValueError(f"Length ({self.len(name)}) of dataset {name} is not equal to length ({len(feature)}) of series {feature_name}")
                self.datasets[name]["data"][feature_name] = feature

    def remove_feature(self, feature_name: str, names=None):

        if names is None or len(names) == 0:
            for name in self.datasets.keys():
                self.datasets[name]["data"].drop([feature_name], axis=1, inplace=True)

        else:
            for name in names:
                self.datasets[name]["data"].drop([feature_name], axis=1, inplace=True)

    def apply_item(self, feature_name: str, item, names=None):
        if names is None or len(names) == 0:
            # add to all datasets
            for name in self.datasets.keys():
                self.datasets[name]["data"][feature_name] = pd.Series([item] * self.len(name))

        else:
            for name in names:
                self.datasets[name]["data"][feature_name] = pd.Series([item] * self.len(name))

    def provide(self, name, y_labels, dtype=None):
        # TODO Out of scope: add the ability to select a range of columns to be x data
        # Flexibility
        dataFrame = self.datasets[name]["data"]
        y = dataFrame[y_labels].values
        x = dataFrame.drop(y_labels, axis=1)
        feature_names = list(x.columns)
        x = x.values

        if dtype:
            x = x.astype(dtype)
            y = y.astype(dtype)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=self.test_size,
                                                            shuffle=True,
                                                            stratify=y)

        return x_train, x_test, y_train, y_test, feature_names

    def get_dataframe(self, name, dtype=None):

        dataFrame = self.datasets[name]["data"]
        dataFrame = dataFrame.values
        dataFrame = dataFrame.astype(dtype)
        return dataFrame


    def len(self, name:str) -> int:
        return len(self.datasets[name]["data"])


if __name__ == "__main__":
    pass
