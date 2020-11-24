
class RenameFeatures:
    def __init__(self, config):
        self.renames = config['renames']

    def __call__(self, datasetComplier):

        datasets = datasetComplier.datasets
        keys = datasets.keys()

        for key in keys:
            datasets[key]['data'].rename(columns=self.renames, inplace=True)



        datasetComplier.datasets = datasets
        datasetComplier.applied_transforms.append("Rename Features")
        return datasetComplier