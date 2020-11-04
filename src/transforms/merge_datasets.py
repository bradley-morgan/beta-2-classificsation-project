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
        self.config = config

    def __call__(self, datasetComplier):

        datasets = datasetComplier.datasets

        merge_all = self.config["merge_all"]
        merge_all_name = self.config["merge_all_name"]
        merge_all_exclude = self.config["merge_all_exclude"]
        leave_original_data = self.config["leave_original_data"]
        groups = self.config["groups"]
        group_names = self.config["group_names"]
        out_dataset = {}

        if len(groups) > 0:
            for group, group_name in zip(groups, group_names):
                group = list(group)
                f_key = group.pop()
                merged_df = datasets[f_key]["data"]

                for df_name in group:
                    data = datasets[df_name]["data"]
                    merged_df = merged_df.append(data, sort=False)

                if leave_original_data:
                    datasets[group_name] = {'data': merged_df}
                else:
                    out_dataset[group_name] = {'data': merged_df}

        if merge_all:
            keys = list(datasets.keys())
            keys = [key for key in keys if key not in merge_all_exclude]

            f_key = keys.pop()
            merged_df = datasets[f_key]["data"]
            for name in keys:
                data = datasets[name]["data"]
                merged_df = merged_df.append(data, sort=False)

            if leave_original_data:
                datasets[merge_all_name] = {'data': merged_df}
            else:
                out_dataset[merge_all_name] = {'data': merged_df}

        if leave_original_data:
            datasetComplier.datasets = datasets
        else:
            datasetComplier.datasets = out_dataset

        return datasetComplier
