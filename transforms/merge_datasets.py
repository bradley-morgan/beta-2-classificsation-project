import pandas as pd

"""
transforms are classes that perform operations and manipulations on datasets
and returns the transformed dataset

All transforms should be called Transform class
transforms must have a __call__ function

Users can define there own custom transforms and define them and their config settings 
when creating & initantiating the dataset class

Merge Datasets Transfrom merges all datasets together into one
"""


class MergeDatasets:

    def __init__(self, config):

        self.merge_all = config["merge_all"]
        self.merge_all_name = config["merge_all_name"]
        self.groups = config["groups"]
        self.group_names = config["group_names"]

    def __call__(self, datasetComplier):

        datasets = datasetComplier.datasets

        out_dataset = {}
        if len(self.groups) > 0:
            for group, group_name in zip(self.groups, self.group_names):
                group = list(group)
                f_key = group.pop()
                merged_df = datasets[f_key]["data"]

                for df_name in group:
                    data = datasets[df_name]["data"]
                    merged_df = merged_df.append(data, sort=False)

                out_dataset[group_name] = {'data': merged_df}
        else:
            out_dataset = datasets

        if self.merge_all:
            j_key = self.group_names.pop()
            merged_df = out_dataset[j_key]["data"]
            for name in self.group_names:
                data = out_dataset[name]["data"]
                merged_df = merged_df.append(data, sort=False,)

            out_dataset[self.merge_all_name] = {'data': merged_df}

        datasetComplier.datasets = out_dataset
        datasetComplier.applied_transforms.append("Merge")
        return datasetComplier
