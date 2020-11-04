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
import itertools

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

    def __init__(self, config, project):

        # User defined attributes
        self.project = project

        self.config = config
        self.name = self.config["name"]
        self.notes = self.config["notes"]
        self.log_data = self.config["log_data"]
        self.src = path(self.config["src"])
        self.stats_config = self.config["stats"]
        self.y_labels = self.config["labels"]
        self.test_size = self.config["test_size"]

        # default attributes
        self.datasets = {}
        self.image_saver = ImageSaver()
        self.run = None
        self.total_features_across_datasets = None
        self.total_shared_features_across_datasets = None
        self.shared_features = None

        # initializer functions
        self.init_run()

    def init_run(self):
        # setup
        if self.log_data:
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
                    raise ValueError(
                        f"Length ({self.len(name)}) of dataset {name} is not equal to length ({len(feature)}) of series {feature_name}")
                self.datasets[name]["data"][feature_name] = feature

        else:
            for name in names:
                if self.len(name) != len(feature):
                    raise ValueError(
                        f"Length ({self.len(name)}) of dataset {name} is not equal to length ({len(feature)}) of series {feature_name}")
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

        # TODO ASSUMPTION WARNING
        warn('Statistics method is currently built on the assumption'
             '\nthat datasets have been merged to contain all classes in one unified data set.'
             'If not please run merge_datasets')

        def find_feature_class(feature_set, feature_type, ag_data_set, ant_data_set):
            feature_list = list(feature_set)
            feature_class_ant = []
            feature_class_ag = []

            for feature in feature_list:
                feature_class_ag.append(True) if feature in ag_data_set else feature_class_ag.append(False)
                feature_class_ant.append(True) if feature in ant_data_set else feature_class_ant.append(False)

                if feature not in ag_data_set or feature not in ant_data_set:
                    ValueError("Value not found in any dataset")

            return {"features": feature_list,
                    "Agonist": feature_class_ag,
                    "antagonist": feature_class_ant,
                    "feature_type": [feature_type] * len(feature_set)}

        def merge_dicts(dict1, dict2):
            return {"features": dict1["features"] + dict2["features"],
                    "Agonist": dict1["Agonist"] + dict2["Agonist"],
                    "antagonist": dict1["antagonist"] + dict2["antagonist"],
                    "feature_type": dict1["feature_type"] + dict2["feature_type"]}


        B2in_data = self.datasets["B2in"]["data"]
        R_data = self.datasets["R"]["data"]
        Z_data = self.datasets["Z"]["data"]


        B2in = set(list(B2in_data.columns))
        R = set(list(R_data.columns))
        Z = set(list(Z_data.columns))

        B2in.remove("target")
        R.remove("target")
        Z.remove("target")

        # TODO MERGED DATASET
        x = list(itertools.chain(B2in, R, Z))
        global_feature_across_datasets = set(x)
        global_feature_across_datasets_len = len(global_feature_across_datasets)

        shared_features_across_datasets = set.intersection(B2in, R, Z)
        shared_features_across_datasets_len = len(shared_features_across_datasets)
        shared_feature_table_across_datasets = {"features": shared_features_across_datasets,
                                                 "feature type": ["shared"] * len(shared_features_across_datasets)}

        unique_features_across_datasets = global_feature_across_datasets.difference(shared_features_across_datasets)
        unique_features_across_datasets_len = len(unique_features_across_datasets)
        unique_feature_table_across_datasets = {"features": unique_features_across_datasets,
                                                 "feature type": ["unique"] * len(unique_features_across_datasets)}

        feature_table_across_datasets = pd.DataFrame.from_dict({
            "features": list(shared_feature_table_across_datasets["features"]) + list(unique_feature_table_across_datasets["features"]),
            "feature_types": list(shared_feature_table_across_datasets["feature type"] + list(unique_feature_table_across_datasets["feature type"]))
        })

        # TODO B2in DATASET
        ag_B2in = set(list(self.datasets["B2in-ag"]["data"].columns))
        ant_B2in = set(list(self.datasets["B2in-ant"]["data"].columns))
        ag_B2in.remove("target")
        ant_B2in.remove("target")
        B2in_shared_class_features = set.intersection(ag_B2in, ant_B2in)
        B2in_unquie_class_features = set.symmetric_difference(ag_B2in, ant_B2in)
        B2in_shared_class_features_len = len(B2in_shared_class_features)
        B2in_unquie_class_features_len = len(B2in_unquie_class_features)

        B2in_unique_dict = find_feature_class(B2in_unquie_class_features, "unique", ag_B2in, ant_B2in)
        B2in_shared_dict = find_feature_class(B2in_shared_class_features, "shared",  ag_B2in, ant_B2in)
        B2in_feature_table = pd.DataFrame.from_dict(merge_dicts(B2in_unique_dict, B2in_shared_dict))

        # TODO R DATASET
        ag_R = set(list(self.datasets["R-ag"]["data"].columns))
        ant_R = set(list(self.datasets["R-ant"]["data"].columns))
        ag_R.remove("target")
        ant_R.remove("target")
        R_shared_class_features = set.intersection(ag_R, ant_R)
        R_unquie_class_features = set.symmetric_difference(ag_R, ant_R)
        R_shared_class_features_len = len(R_shared_class_features)
        R_unquie_class_features_len = len(R_unquie_class_features)

        R_unique_dict = find_feature_class(R_unquie_class_features, "unique", ag_R, ant_R)
        R_shared_dict = find_feature_class(R_shared_class_features, "shared",  ag_R, ant_R)
        R_feature_table = pd.DataFrame.from_dict(merge_dicts(R_unique_dict, R_shared_dict))

        # TODO Z DATASET
        ag_Z = set(list(self.datasets["Z-ag"]["data"].columns))
        ant_Z = set(list(self.datasets["Z-ant"]["data"].columns))
        ag_Z.remove("target")
        ant_Z.remove("target")
        Z_shared_class_features = set.intersection(ag_Z, ant_Z)
        Z_unique_class_features = set.symmetric_difference(ag_Z, ant_Z)
        Z_shared_class_features_len = len(Z_shared_class_features)
        Z_unique_class_features_len = len(Z_unique_class_features)

        Z_unique_dict = find_feature_class(Z_unique_class_features, "unique", ag_Z, ant_Z)
        Z_shared_dict = find_feature_class(Z_shared_class_features, "shared",  ag_Z, ant_Z)
        Z_feature_table = pd.DataFrame.from_dict(merge_dicts(Z_unique_dict, Z_shared_dict))

        # Calculate class balance

        # Log
        sns.set()
        self.run.log({'Feature Size Across all Data-sets (B2in, R, Z)': global_feature_across_datasets_len})
        # self.run.log({f'{name} Feature Size': dataset["feature_size"]})

        # Features Plot
        plt.bar(
            x=['shared', 'unique'],
            height=[shared_features_across_datasets_len, unique_features_across_datasets_len],
            width=0.8
        )
        plt.title(
            f'Features Shared Across all Classes(Ag, Ant) & Data-sets (B2in, R, Z): Total Features={global_feature_across_datasets_len}')
        plt.ylabel('Counts')
        plt.xlabel('Feature Types')
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name='Feature Size Across all Data-sets (B2in, R, Z',
                              format='png')
        plt.clf()

        self.run.log({'Shared Features Size Between Antagonist and Agonist for B2in': B2in_shared_class_features_len})
        self.run.log({'Unique Features Size Between Antagonist and Agonist for B2in': B2in_unquie_class_features_len})

        plt.bar(
            x=["shared", "unique"],
            height=[B2in_shared_class_features_len, B2in_unquie_class_features_len],
            width=0.8
        )
        plt.title(f'Features Shared Between Antagonist and Agonist for B2in')
        plt.ylabel("Counts")
        plt.xlabel("Feature Types")
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name='Plot: Features Shared Between Antagonist and Agonist for B2in',
                              format='png')
        plt.clf()

        plt.bar(
            x=["shared", "unique"],
            height=[R_shared_class_features_len, R_unquie_class_features_len],
            width=0.8
        )
        plt.title(f'Features Shared Between Antagonist and Agonist for R')
        plt.ylabel("Counts")
        plt.xlabel("Feature Types")
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name='Plot: Features Shared Between Antagonist and Agonist for R',
                              format='png')
        plt.clf()

        plt.bar(
            x=["shared", "unique"],
            height=[Z_shared_class_features_len, Z_unique_class_features_len],
            width=0.8
        )
        plt.title(f'Features Shared Between Antagonist and Agonist for Z')
        plt.ylabel("Counts")
        plt.xlabel("Feature Types")
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name='Plot: Features Shared Between Antagonist and Agonist for Z',
                              format='png')
        plt.clf()

        # Plot Tables

        self.run.log({'Feature Table Across All Datasets (B2in, Z, R) ': wandb.Table(dataframe=feature_table_across_datasets)})
        self.run.log({'Feature Table B2in': wandb.Table(dataframe=B2in_feature_table)})
        self.run.log({'Feature Table R': wandb.Table(dataframe=R_feature_table)})
        self.run.log({'Feature Table Z': wandb.Table(dataframe=Z_feature_table)})



    def log(self):
        if not self.log_data:
            warn("Log Warning: Log has not run and is set to False thus no run object is available for logging")
            return

        sns.set()
        # Use .loc[0] for ant and .loc[1] for ag when getting class balance from series
        df_names = self.stats_config["names"]

        if df_names is None or len(df_names) == 0:
            # Loop over all datasets
            df_names = self.datasets.keys()

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

    def terminate(self):
        self.run.finish()

    def len(self, name: str) -> int:
        return len(self.datasets[name]["data"])


if __name__ == "__main__":
    pass
