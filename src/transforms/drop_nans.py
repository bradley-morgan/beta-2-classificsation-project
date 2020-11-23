
class DropNans:

    def __init__(self, config):
        self.config = config

    def __call__(self, datasetCompiler):

        datasets = datasetCompiler.datasets
        target_datasets = self.config["target_datasets"]

        if target_datasets is None or len(target_datasets) == 0:
            # Apply to all datasets
            keys = list(datasets.keys())

            for name in keys:
                datasets[name]["data"].dropna(axis=1, inplace=True)
                if datasets[name]["data"].isna():
                    raise ValueError('drop_nans.py Failed to remove all NaN values')

        else:
            for name in target_datasets:
                datasets[name]["data"].dropna(axis=1, inplace=True)
                if datasets[name]["data"].isna().values.any():
                    raise ValueError('drop_nans.py Failed to remove all NaN values')

        datasetCompiler.datasets = datasets
        datasetCompiler.applied_transforms.append("drop_nans")
        return datasetCompiler