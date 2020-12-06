import pandas as pd
import os
from tools.remove import remove
from tools.set_path import path
from sklearn.model_selection import train_test_split
from tools.anonymousClass import Obj
from transforms.merge_datasets import MergeDatasets
from transforms.change_nans import ChangeNans
from transforms.clean_feature_names import CleanFeatureNames
from transforms.remove_features import RemoveFeatures
from transforms.rename_feature import RenameFeatures
from configs import get
import pickle



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

    def provide(self, name, dtype=None):
        # TODO Out of scope: add the ability to select a range of columns to be x data
        # Flexibility
        dataFrame = self.datasets[name]["data"]
        y = dataFrame[self.y_labels].to_numpy()
        x = dataFrame.drop(self.y_labels, axis=1)
        feature_names = x.columns.to_numpy()
        x = x.to_numpy()

        if dtype:
            x = x.astype(dtype)
            y = y.astype(dtype)

        x_train, x_hold_out, y_train, y_hold_out = train_test_split(x, y,
                                                            test_size=self.test_size,
                                                            shuffle=True,
                                                            stratify=y)

        return Obj(
                    x_train=x_train,
                    x_hold_out=x_hold_out,
                    y_train=y_train,
                    y_hold_out=y_hold_out,
                    feature_names=feature_names
        )
        # return x_train, x_hold_out, y_train, y_hold_out, feature_names

    def get_dataframe(self, name, dtype=None):

        dataFrame = self.datasets[name]["data"]
        dataFrame = dataFrame.values
        dataFrame = dataFrame.astype(dtype)
        return dataFrame


    def len(self, name:str) -> int:
        return len(self.datasets[name]["data"])


    def save_as_pickle(self, data, name: str, dest: str):
        if not os.path.exists(dest):
            os.mkdir(dest)

        if not name.endswith('.pickle'):
            name = f'{name}.pickle'

        with open(os.path.join(dest, name), 'wb') as out_put_file:
            pickle.dump(data, out_put_file)
            print(f'Pickle Save Successful for: {dest}')

    @staticmethod
    def load_from_pickle(file_path):

        if not os.path.exists(file_path):
            raise FileExistsError(f'File Does not exist: {file_path} try using tools.path')


        with open(file_path, 'rb') as input_put_file:
            data = pickle.load(input_put_file)
            print(f'Pickle Load Successful for: {file_path}')

        return data


if __name__ == "__main__":

    filter_name = 'merged-B2in-Z-R'
    filter_groups = [('B2in-ant', 'B2in-ag'), ('R-ant', 'R-ag'), ('Z-ant', 'Z-ag')]
    filter_group_names = ['b2in', 'R', 'Z']
    filter_remove_feature = 'Ligand_Pose'
    filtered_clean_feature = 'target'


    non_filter_name = 'merged-3sn6-4lde-5jqh'
    non_filter_groups = [('3sn6-ant', '3sn6-ag'), ('4lde-ant', '4lde-ag'), ('5jqh-ant', '5jqh-ag')]
    non_filter_group_names = ['3sn6', '4lde', '5jqh']
    non_filter_remove_feature = 'Ligand_Pose2'
    non_filtered_clean_feature = 'Actions'


    config = Obj(
        project_name='b2ar-filter-rfc-optimisation',
        src='data/filtered',
        y_labels='target',
        test_size=0.02,
        notes='This is test script for rfc sweeps',
        clean_features=dict(
            exceptions=[filtered_clean_feature] # For non-filtered = Action
        ),
        rename_features=dict(
            renames=dict(Action='target')
        ),
        merge=dict(
            merge_all=True,
            merge_all_name=filter_name,
            groups=filter_groups,
            group_names=filter_group_names
        ),
        remove_features=dict(
            search_params=["Clash", "Proximal"]
        ),
        change_nans=dict(
            value=0
        ),
    )

    # Data Preprocessing
    data_sets = DatasetCompiler(src=config.src, y_labels=config.y_labels, test_size=config.test_size)
    data_sets.load()
    # Apply only for filtered data
    data_sets.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
    data_sets.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
    data_sets.remove_feature(feature_name=filter_remove_feature)
    data_sets = CleanFeatureNames(config.clean_features)(data_sets)
    #data_sets = RenameFeatures(config.rename_features)(data_sets)
    data_sets = RemoveFeatures(config.remove_features)(data_sets)
    data_sets = MergeDatasets(config.merge)(data_sets)
    data_sets = ChangeNans(config.change_nans)(data_sets)
    data = data_sets.provide(filter_name, 'int64')
    data_sets.save_as_pickle(data, dest='../data/processed/filtered', name='dataset1.pickle')

    # sm_data = DatasetCompiler.load_from_pickle('./data/processed/clean_data.pickle')
    # data = DatasetCompiler.load_from_pickle('./data/processed/lrg_clean_data.pickle')
    # import numpy as np
    # vals, counts = np.unique(data.y_hold_out, return_counts=True)

