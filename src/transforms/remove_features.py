class RemoveFeatures:

    def __init__(self, config):
        self.search_params = config["search_params"]

    def __call__(self, datasetCompiler):

        keys = datasetCompiler.datasets.keys()
        search_params = self.search_params

        for key in keys:
            data = datasetCompiler.datasets[key]["data"]
            cols = []
            for col in data.columns:
                finds = []
                for search_param in search_params:
                    if search_param not in col:
                        finds.append(False)
                    else:
                        finds.append(True)

                if not any(finds):
                    cols.append(col)

            datasetCompiler.datasets[key]["data"] = data[cols]
        datasetCompiler.applied_transforms.append("Remove_features")

        return datasetCompiler