
class ChangeNans:

    def __init__(self, config):
        self.value = config['value']

    def __call__(self, datasetCompiler):

        datasets = datasetCompiler.datasets

        keys = list(datasets.keys())

        for name in keys:
            datasets[name]["data"].fillna(self.value, inplace=True)

        datasetCompiler.datasets = datasets
        datasetCompiler.applied_transforms.append("Change_Nans")
        return datasetCompiler
