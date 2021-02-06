import pandas as pd


class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, datasets: dict):

        if self.config.keys() is None:
            return datasets

        for item in self.config.keys():
            params = self.config[item]
            name = params['name']
            item_to_apply = params['item']
            apply_to = params['apply_to']

            for dataset_name in apply_to:
                datasets[dataset_name]['data'][name] = pd.Series([item_to_apply] * len(datasets[dataset_name]['data']))

        return datasets