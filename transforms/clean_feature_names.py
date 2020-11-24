
class CleanFeatureNames:

    def __init__(self, config):
        self.exceptions = config['exceptions']

    def __call__(self, datasetComplier):

        datasets = datasetComplier.datasets
        keys = datasets.keys()

        for key in keys:

            for col_name in datasets[key]["data"].columns:

                if col_name in self.exceptions:
                    break

                new_col_names = col_name
                isLetterA = new_col_names.startswith("A")

                if isLetterA:
                    new_col_names = new_col_names[3:]


                else:
                    new_col_names = new_col_names[2:]

                if new_col_names.startswith("0"):
                    new_col_names = new_col_names[1:]

                datasets[key]["data"].rename({col_name: new_col_names}, axis=1, inplace=True)

        datasetComplier.datasets = datasets
        datasetComplier.applied_transforms.append("Clean Features")
        return datasetComplier



