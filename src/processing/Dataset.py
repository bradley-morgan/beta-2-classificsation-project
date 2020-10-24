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
        self.project = config["project"]
        self.wandb_key = config["wandb_key"]

        self.config = config["dataset"]
        self.name = self.config["name"]
        self.notes = self.config["notes"]
        self.log = self.config["log"]
        self.src = path(self.config["src"])
        self.transform_config = self.config["transforms"]
        self.stats_config = self.config["stats"]
        self.y_labels = self.config["labels"]
        self.test_size = self.config["test_size"]


        # default attributes
        self.datasets = {}
        self.transforms = []
        self.transforms_flag = False
        self.image_saver = ImageSaver()

        # Hidden attributes
        self.__transform_dir__ = path('library/transforms')

        # initializer functions
        self.init_run()
        self.__init_transforms__()

    def init_run(self):
        # setup
        if self.log:
            self.run = wandb.init(project=self.project,
                                  name=self.name,
                                  id='dataset1011001',
                                  notes=self.notes,
                                  config=self.config,
                                  reinit=True,
                                  )


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
        self.transforms_flag = True

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


    def statistics(self):

        #TODO ASSUMPTION WARNING
        warn('Statistics method is currently built on the assumption'
             '\nthat datasets have been merged to contain all classes in one unified data set.'
             'If not please run merge_datasets')

        df_names = self.stats_config["names"]
        label_feature_name = self.stats_config["label_feature_name"]

        if df_names is None or len(df_names) == 0:
            # Loop over all datasets
            df_names = self.datasets.keys()

        print('\nCalculating Data-set Statistics')
        with tqdm(total=len(df_names), colour='blue', bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}') as progress_bar:
            for name in df_names:
                dataset = self.datasets[name]["data"]
                feature_names = list(dataset.columns)
                feature_names.remove(label_feature_name)

                # Sample size
                sample_size = len(dataset)

                # Feature_size
                feature_size = len(feature_names)

                #Class balance
                class_balance = dataset[label_feature_name].value_counts()
                class_names = list(class_balance.index)

                # shared and unique features table 2D array - feature name, classes
                # On a feature level individual samples within each feature might have shared features
                # This is calculated on the assumption that columns with nans means the feature didnt
                # exist when being merged
                feature_table = {"feature_name": []}

                for feature in feature_names:

                    feature_table["feature_name"].append(feature)
                    feature_col = dataset[[feature, label_feature_name]]

                    for class_name in class_names:

                        class_feature = feature_col[feature_col[label_feature_name] == class_name]
                        is_nan = class_feature.isna().values.any()

                        # Flip isNan logic to help understanding. if No Nans detected (False)
                        # then that means this feature is present for the current class thus (True)
                        is_class_feature = False if is_nan else True

                        # check if class name exist if so append if not create then append
                        try:
                            feature_table[str(class_name)].append(is_class_feature)
                        except KeyError:
                            feature_table[str(class_name)] = []
                            feature_table[str(class_name)].append(is_class_feature)

                # Convert to Pd frame this aint effecient but its quickest way i could think of Feature Table
                feature_table = pd.DataFrame.from_dict(feature_table)
                # Shared Features Histogram
                tmp_feature_table = feature_table.drop(["feature_name"], axis=1)

                shared_features = []
                for classes in tmp_feature_table.itertuples(index=True):
                    classes = list(classes)

                    if all(classes):
                        shared_features.append('shared')
                    else:
                        shared_features.append('not-shared')

                shared = shared_features.count('shared')
                not_shared = shared_features.count('not-shared')

                shared_features = pd.Series(data=shared_features)
                feature_table["Is Shared"] = shared_features

                self.datasets[name]["sample_size"] = sample_size
                self.datasets[name]["feature_size"] = feature_size
                self.datasets[name]["class_balance"] = class_balance
                self.datasets[name]["feature_table"] = feature_table
                self.datasets[name]["shared_features"] = shared
                self.datasets[name]["not_shared_features"] = not_shared

                progress_bar.update(1)

    def log(self):
        if not self.log:
            raise ValueError("Log Error: Log is set to False thus no run object is available for logging")

        sns.set()
        # Use .loc[0] for ant and .loc[1] for ag when getting class balance from series
        df_names = self.stats_config["names"]

        if df_names is None or len(df_names) == 0:
            # Loop over all datasets
            df_names = self.datasets.keys()

        self.run.log({f'Transforms Applied': list(self.transform_config.keys())})

        for name in df_names:
            dataset = self.datasets[name]

            self.run.log({f'{name} Sample Size': dataset["sample_size"]})
            self.run.log({f'{name} Feature Size': dataset["feature_size"]})

            # Class Balance Plot
            class_names = list(dataset["class_balance"].index)
            height = dataset["class_balance"].values
            plt.bar(
                x=class_names,
                height=height,
                width=0.8
            )
            plt.title(f'{name} Class Balance')
            plt.ylabel('Counts')
            plt.xlabel('Classes')
            plt.xticks(class_names)
            self.image_saver.save(plot=plt.gcf(),
                                  run=self.run,
                                  name=f'{name} Class Balance',
                                  format='png')
            plt.clf()

            plt.bar(
                x=['shared', 'not-shared'],
                height=[dataset["shared_features"], dataset["not_shared_features"]],
                width=0.8
            )
            plt.title(f'{name}: Number of features shared and not shared across all classes')
            plt.ylabel('Counts')
            plt.xlabel('Feature type')
            plt.xticks(class_names)
            self.image_saver.save(plot=plt.gcf(),
                                  run=self.run,
                                  name=f'{name} feature shared',
                                  format='png')
            plt.clf()
            
            wandb.log({f'{name} Feature - Class Table': wandb.Table(dataframe=dataset["feature_table"])})

    def terminate(self):
        self.run.finish()

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
