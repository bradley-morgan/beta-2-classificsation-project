import pandas as pd
import os
import importlib
from src.utils.remove import remove
from src.utils.set_path import path


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


class Dataset():

    def __init__(self, config:dict):

        # User defined attributes
        self.src = path(config["src"])
        self.transform_config = config["transforms"]
        self.y_labels = config["labels"]


        # default attributes
        self.datasets = {}
        self.transforms = []

        # Hidden attributes
        self.__transform_dir__ = path('lib/transforms')

        # initializer functions
        self.__init_transforms__()

    def load(self):

        folder_list = os.listdir(self.src)
        # Handle .DS_store for mac
        folder_list = remove(folder_list, '.DS_Store')

        for folder in folder_list:
            file_list = os.listdir(os.path.join(self.src, folder))

            file_list = remove(file_list, '.DS_Store')
            for file in file_list:
                data = pd.read_csv(os.path.join(self.src, folder, file))
                file_name = file.split('.')
                dataset_name = f'{folder}-{file_name[0]}'

                self.datasets[dataset_name] = {'data': data}

    def save(self):
        pass

    def transform(self):
        for transform in self.transforms:
            self.datasets = transform(self.datasets)

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

    def remove_feature(self, feature_name:str, names=None):

        if names is None or len(names) == 0:
            for name in self.datasets.keys():
                self.datasets[name]["data"].drop([feature_name], axis=1, inplace=True)

        else:
            for name in names:
                self.datasets[name]["data"].drop(feature_name, axis=1, inplace=True)

    def apply_item(self, feature_name: str, item, names=None):
        if names is None or len(names) == 0:
            # add to all datasets
            for name in self.datasets.keys():
                self.datasets[name]["data"][feature_name] = pd.Series([item] * self.len(name))

        else:
            for name in names:
                self.datasets[name]["data"][feature_name] = pd.Series([item] * self.len(name))


    def provide(self, name, y_labels=None, shuffle=False, dtype=None):
        # TODO Out of scope: add the ability to select a range of columns to be x data
        # Flexibility
        dataFrame = self.datasets[name]["data"]
        if shuffle:
            dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

        if y_labels:
            y = dataFrame[y_labels].values
            x = dataFrame.drop(y_labels, axis=1)
            feature_names = list(x.columns)
            x = x.values

            if dtype:
                x = x.astype(dtype)
                y = y.astype(dtype)

            return x, y, feature_names
        else:
            dataFrame = dataFrame.values
            dataFrame = dataFrame.astype(dtype)
            return dataFrame

    def statistics(self, kind:str):
        pass

    def get_items(self):
        pass

    def log(self):
        pass

    def len(self, name:str) -> int:
        return len(self.datasets[name]["data"])

    def __init_transforms__(self):
        """
        Checks that user-set transforms exist as files in the transform folder
        If it exists it then load and initializes the transform
        :return:
        """
        transform_files = os.listdir(self.__transform_dir__)
        transform_files = [file for file in transform_files if file.endswith('.py')]

        user_transform_files = self.transform_config.keys()

        transforms = []
        for user_transform in user_transform_files:
            if f'{user_transform}.py' in transform_files:
                plugin = importlib.import_module(f'{user_transform}', package=self.__transform_dir__)
                plugin = plugin.Transform(self.transform_config[user_transform])
                transforms.append(plugin)

        self.transforms = transforms


if __name__ == "__main__":

    config = {
        'project': 'wandb project name',
        'wandb_key': '003bdcde0a7e623fdeb0425c3079a7aed09a32e6',

        'dataset': {
            'src': '../data/raw',
            'labels': 'target',
            'transforms': {
                'merge_datasets': dict(
                    name='beta-2-ag-ant',
                ),
                'change_nans': dict(
                    value=0
                )
            }
        },
        'models': {
            'random_forest': dict(
                run_name='RFC Test Run',
                model_name="Standard RandomForest 1",
                dataset="beta-2-ag-ant",
                y_labels="target",
                k_folds=10,
                learning_curve=True,
                n_estimators=10,
                max_features=100
            )
        }
    }

    dataset = Dataset(config=config["dataset"])
    dataset.load()
    dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
    dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
    dataset.remove_feature(feature_name='Ligand_Pose', names=['R-ag', 'B2in-ant'])
    dataset.transform()
    a = 0


    # feature_ag = pd.Series([1] * dataset.len('R-ag'))
    # feature_ant = pd.Series([0] * dataset.len('R-ag'))
    #
    #dataset.add_feature(feature_name="target", feature=feature_ag, names=['R-ag'], )
    # a = 0
