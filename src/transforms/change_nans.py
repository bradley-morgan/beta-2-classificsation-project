
class ChangeNans:

    def __init__(self, config):
        self.config = config

    def __call__(self, datasetCompiler):

        datasets = datasetCompiler.datasets

        keys = list(datasets.keys())

        for name in keys:
            datasets[name]["data"].fillna(self.config['value'], inplace=True)

        datasetCompiler.datasets = datasets
        return datasetCompiler
