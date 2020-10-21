
class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, datasets: dict):
        keys = list(datasets.keys())

        for name in keys:
            datasets[name]["data"].fillna(self.config['value'], inplace=True)

        return datasets
